"""Standalone AutoQuant script for ESP-Detection.

Quantize an already-trained ESPDet model into a deployable ``.espdl`` model
without re-running the full ``espdet_run.py`` pipeline.

It uses esp-ppq's ``espdl_auto_quantize_onnx`` to automatically *search* for a
good quantization configuration (calibration algorithm, mixed precision, ...)
instead of using one fixed config. See the reference samples in
``esp_ppq/samples/AutoQuant``.

Input may be:
  * a trained ``.pt`` -> it is first exported to ONNX (ESPDet head -> box/score
    outputs, opset 13) via ``deploy/export.Export``;
  * an already-exported ``.onnx`` -> quantized directly.

Ranking of candidate configs:
  * default: esp-ppq's built-in graph-wise SNR error (no validation set needed,
    only calibration images);
  * ``--data <dataset.yaml>``: rank by detection mAP on the val split, reusing
    the quantized-model validator in ``deploy/eval_quantized_model.py``.

Example::

    uv run python deploy/auto_quantize.py \
      --model runs/train-8/weights/best.pt \
      --size 224 224 \
      --target esp32p4 \
      --calib_data path/to/number_calib \
      --espdl espdet_pico_224_224_number.espdl \
      --num_candidates 5
"""

import argparse
import os

import onnx
import torch
from onnxsim import simplify
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from esp_ppq.api import AutoQuantSearchSetting, espdl_auto_quantize_onnx

from deploy.export import Export


class CalibDataset(Dataset):
    """Calibration dataset whose preprocessing mirrors training and on-device inference.

    Uses *letterbox* (aspect-preserving resize + centered gray padding) instead of a
    plain stretch-resize, then scales to [0, 1] RGB. This matches the ESP-DL
    ``ImagePreprocessor(model, {0,0,0}, {255,255,255})`` + ``enable_letterbox({114,114,114})``
    used on the chip. Calibrating on the same input distribution the model actually
    sees at deploy time is critical for 8-bit quantization accuracy.
    """

    def __init__(self, path, img_shape=224, pad_value=114):
        super().__init__()
        self.height, self.width = img_shape if isinstance(img_shape, (list, tuple)) else (img_shape, img_shape)
        self.pad_value = pad_value
        self.to_tensor = transforms.ToTensor()  # HWC uint8 [0,255] -> CHW float [0,1]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Calibration dir not found: {path}")
        self.imgs_path = [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not self.imgs_path:
            raise FileNotFoundError(f"No images (.jpg/.jpeg/.png/.bmp) found under {path}")

    def _letterbox(self, img):
        """Resize keeping aspect ratio, then center-pad to (width, height) with pad_value."""
        iw, ih = img.size  # PIL size is (w, h)
        scale = min(self.width / iw, self.height / ih)
        nw, nh = round(iw * scale), round(ih * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), (self.pad_value,) * 3)
        canvas.paste(img, ((self.width - nw) // 2, (self.height - nh) // 2))
        return canvas

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert("RGB")
        return self.to_tensor(self._letterbox(img))


def prepare_onnx(model_path, size):
    """Return an ONNX path ready for quantization, exporting from .pt if needed."""
    if model_path.endswith(".pt"):
        print("\033[32mExport .pt -> ONNX (ESPDet head, opset 13)\033[0m")
        Export(model_path, size)  # writes best.onnx next to best.pt
        onnx_path = model_path[: -len(".pt")] + ".onnx"
    elif model_path.endswith(".onnx"):
        onnx_path = model_path
    else:
        raise ValueError(f"--model must end with .pt or .onnx, got: {model_path}")

    # Simplify + shape-infer in place (same as deploy/quantize.py).
    model = onnx.load(onnx_path)
    model, ok = simplify(model)
    assert ok, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_path)
    return onnx_path


def build_evaluate_fn(model_path, data_yaml, size, device):
    """Build an mAP-based evaluate_fn reusing the repo's quantized-model validator.

    Requires a .pt (for the YOLO val pipeline) and a dataset yaml with a val split.
    Returns (score=mAP50-95, extras={mAP50, mAP50-95}); higher is better.
    """
    from esp_ppq.executor import TorchExecutor
    from ultralytics import YOLO

    from deploy.eval_quantized_model import make_quant_validator_class

    if not model_path.endswith(".pt"):
        raise ValueError("--data (mAP eval) needs the trained .pt as --model, not an .onnx")

    # val requires an integer imgsz; for square inputs pass the side length.
    val_imgsz = size[0] if size[0] == size[1] else size
    # Load the YOLO model once and reuse it across candidates (avoids reloading the
    # model and leaking GPU memory on every AutoQuant candidate evaluation).
    model = YOLO(model_path)

    def evaluate_fn(quant_graph):
        executor = TorchExecutor(graph=quant_graph, device=device)
        QuantValidator = make_quant_validator_class(executor)
        try:
            results = model.val(
                data=data_yaml,
                split="val",
                imgsz=val_imgsz,
                device=device,
                validator=QuantValidator,
                rect=False,
                save_json=False,
                plots=False,
            )
            map50 = float(results.box.map50)
            map5095 = float(results.box.map)
            return map5095, {"mAP50": map50, "mAP50-95": map5095}
        finally:
            # Release the per-candidate quantized-graph executor and free GPU memory
            # so it doesn't accumulate across the many AutoQuant candidates (OOM guard).
            del executor, QuantValidator
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    return evaluate_fn


def auto_quantize(
    model_path,
    size,
    target,
    calib_data,
    espdl,
    num_candidates=5,
    search_mode="fast",
    device=None,
    batchsz=32,
    calib_steps=32,
    run_dir="outputs/auto_quant",
    data_yaml=None,
    resume=False,
    aggressive=False,
):
    assert isinstance(size, list) and len(size) == 2, "size should be a list [h, w]."
    if device is None:
        # device_count() (not is_available()): some containers report
        # is_available()==True with 0 usable GPUs (e.g. CUDA_VISIBLE_DEVICES="").
        device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    h, w = size
    input_shape = [3, h, w]  # AutoQuant input_shape excludes the batch dim

    onnx_path = prepare_onnx(model_path, size)

    dataloader = DataLoader(
        dataset=CalibDataset(calib_data, img_shape=size),
        batch_size=batchsz,
        shuffle=False,
    )

    def collate_fn(batch):
        return batch.to(device)

    # mAP-based ranking is opt-in (needs a val set); otherwise rank by SNR error.
    if data_yaml is not None:
        evaluate_fn = build_evaluate_fn(model_path, data_yaml, size, device)
        score_direction = "maximize"  # higher mAP is better
    else:
        evaluate_fn = None
        score_direction = "minimize"  # built-in SNR error: lower is better

    setting = AutoQuantSearchSetting(
        search_mode=search_mode,
        num_of_candidates=num_candidates,
        score_direction=score_direction,
        run_dir=run_dir,
        resume=resume,
    )
    # ESP-DL on-chip parity (CRITICAL): force scale alignment across the graph.
    # The default (force_overlap=False) lets connected tensors keep mismatched
    # power-of-2 scales, inserting many RequantizeLinear ops. Those are lossless in
    # the esp-ppq simulator (high sim mAP) but lossy on the chip, accumulating
    # rounding error that collapses detection scores on-device. Forcing alignment
    # cuts RequantizeLinear (e.g. 19 -> ~1) so simulator and chip agree.
    setting.param_space["fusion_alignment"]["force_overlap"] = [True]
    if search_mode == "fast":
        setting.top_strategy = 10
        setting.early_stop_patience = 5
    if not aggressive:
        # DEFAULT = chip-faithful recipe. Disable int16 (mixed_precision) and the
        # gradient/reconstruction tuning (tqt, blockwise_reconstruction). For this
        # model these add ~0 mAP over plain calibration + bias_correction, but they
        # introduce sim<->chip divergence (int16<->int8 boundaries; tuning that
        # overfits the esp-ppq float simulator), risking collapsed scores on-device.
        # Combined with the forced scale alignment above, this matches what was
        # verified working on-chip. Use --aggressive to also search these.
        for k in ("mixed_precision", "tqt", "blockwise_reconstruction"):
            setting.strategy_space[k] = [False]
    else:
        setting.strategy_space["mixed_precision"] = [True, False]

    os.makedirs(os.path.dirname(espdl) or ".", exist_ok=True)
    topk = espdl_auto_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=input_shape,
        evaluate_fn=evaluate_fn,
        target=target,
        collate_fn=collate_fn,
        setting=setting,
        device=device,
        verbose=0,
    )

    print("\n\033[32m=== Top-K candidates ===\033[0m")
    for c in topk:
        score = c.get("score")
        extra = {k: v for k, v in c.items() if k not in ("score", "hash", "index", "files", "strategy", "params")}
        print(f"#{c.get('index'):04d}  score={score}  folder={c.get('folder')}  {extra}")
    print(f"\n\033[32mBest model exported to: {espdl}\033[0m")
    return topk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone AutoQuant: quantize a trained ESPDet model to .espdl."
    )
    parser.add_argument("--model", type=str, required=True, help="Trained model: .pt (auto-exported) or .onnx")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Input resolution [h w]")
    parser.add_argument("--target", type=str, default="esp32p4", help="Target chip: 'esp32p4' or 'esp32s3'")
    parser.add_argument("--calib_data", type=str, required=True, help="Calibration image directory")
    parser.add_argument("--espdl", type=str, required=True, help="Output .espdl path (best candidate)")
    parser.add_argument("--num_candidates", type=int, default=5, help="Top-K candidates to keep")
    parser.add_argument("--search_mode", type=str, default="fast", choices=["fast", "exhaustive"],
                        help="AutoQuant search mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() > 0 else "cpu",
                        help="'cpu' or 'cuda'")
    parser.add_argument("--batchsz", type=int, default=32, help="Calibration batch size")
    parser.add_argument("--calib_steps", type=int, default=32, help="Number of calibration batches")
    parser.add_argument("--run_dir", type=str, default="outputs/auto_quant", help="AutoQuant results dir")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional dataset yaml to rank candidates by val mAP (needs a .pt model)")
    parser.add_argument("--resume", action="store_true", help="Resume search from an existing run_dir")
    parser.add_argument("--aggressive", action="store_true",
                        help="Also search int16 mixed_precision + tqt/blockwise reconstruction. "
                             "Off by default: those add ~0 mAP here and risk sim<->chip divergence. "
                             "If you enable it, re-verify the model on the actual chip.")

    args = parser.parse_args()
    auto_quantize(
        model_path=args.model,
        size=args.size,
        target=args.target,
        calib_data=args.calib_data,
        espdl=args.espdl,
        num_candidates=args.num_candidates,
        search_mode=args.search_mode,
        device=args.device,
        batchsz=args.batchsz,
        calib_steps=args.calib_steps,
        run_dir=args.run_dir,
        data_yaml=args.data,
        resume=args.resume,
        aggressive=args.aggressive,
    )

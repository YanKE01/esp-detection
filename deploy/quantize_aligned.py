"""Fixed-config, scale-ALIGNED int8 quantization for ESP-DL.

Motivation: AutoQuant-produced .espdl models had many RequantizeLinear ops (e.g. 19)
because connected tensors ended up with mismatched power-of-2 scales, while a known
on-chip-working reference model had only 3. The extra rescales / scale misalignment
are the leading suspect for "high simulator mAP but collapsed detection scores on the
actual chip".

This script does a SINGLE deterministic quantization (no AutoQuant search) with
aggressive scale alignment so far fewer RequantizeLinear ops are produced:
  - calib_algorithm = percentile (the winning calibration for this model)
  - bias_correction on; NO tqt / blockwise / mixed_precision (chip-faithful)
  - fusion_setting.force_alignment_overlap = True  (propagate scale alignment upstream)
  - align elementwise/concat/resize/pooling aggressively

It then reports the RequantizeLinear count of the exported model (verify it dropped
toward the reference's 3 BEFORE flashing) and, if --data is given, the simulator mAP.

Example:
    CUDA_VISIBLE_DEVICES=0 uv run python -m deploy.quantize_aligned \
      --model runs/train-8/weights/best.pt --size 224 224 --target esp32p4 \
      --calib_data /root/gpufree-data/number/test/images \
      --espdl espdet_pico_224_224_number_aligned.espdl \
      --data cfg/datasets/number.yaml
"""

import argparse
import os
from collections import Counter

import onnx
import torch
from onnxsim import simplify
from torch.utils.data import DataLoader

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx

from deploy.auto_quantize import CalibDataset, build_evaluate_fn, prepare_onnx


def count_requantize(espdl_path):
    """Parse an exported .espdl and return a histogram of node op types."""
    from esp_ppq.parser.espdl.FlatBuffers.Dl import Model

    buf = open(espdl_path, "rb").read()
    off = 16 if buf[:4] == b"EDL2" else 0
    g = Model.Model.GetRootAs(buf, off).Graph()
    c = Counter()
    for i in range(g.NodeLength()):
        n = g.Node(i)
        c[n.OpType().decode() if n.OpType() else "?"] += 1
    return c


def build_aligned_setting():
    """espdl setting tuned for minimal RequantizeLinear / maximal scale alignment."""
    s = QuantizationSettingFactory.espdl_setting()
    # winning calibration for this model (percentile >> kl on the number model).
    # Only activations: weights are per-channel and the percentile observer rejects
    # per-channel, so leave parameter calibration at the espdl default.
    s.quantize_activation_setting.calib_algorithm = "percentile"
    # chip-faithful: only bias correction, no gradient/reconstruction tuning
    s.bias_correct = True
    s.tqt_optimization = False
    s.blockwise_reconstruction = False
    s.equalization = False
    # aggressive scale alignment -> fewer RequantizeLinear (the whole point)
    s.fusion = True
    s.fusion_setting.align_quantization = True
    s.fusion_setting.force_alignment_overlap = True
    s.fusion_setting.align_elementwise_to = "Align to Large"
    s.fusion_setting.align_concat_to = "Align to Output"
    s.fusion_setting.align_resize_to = "Align to Output"
    s.fusion_setting.align_avgpooling_to = "Align to Output"
    return s


def quantize_aligned(model_path, size, target, calib_data, espdl,
                     device=None, batchsz=16, calib_steps=16, data_yaml=None):
    if device is None:
        device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    h, w = size
    onnx_path = prepare_onnx(model_path, size)

    dataloader = DataLoader(CalibDataset(calib_data, img_shape=size), batch_size=batchsz, shuffle=False)

    def collate_fn(batch):
        return batch.to(device)

    setting = build_aligned_setting()
    os.makedirs(os.path.dirname(espdl) or ".", exist_ok=True)
    graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=[1, 3, h, w],
        target=target,
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=setting,
        device=device,
        error_report=False,
        skip_export=False,
        verbose=0,
    )

    ops = count_requantize(espdl)
    print("\n\033[32m=== exported op histogram ===\033[0m")
    for k, v in sorted(ops.items()):
        print("   %3d  %s" % (v, k))
    print("\033[32mRequantizeLinear = %d  (reference on-chip-working model has 3)\033[0m"
          % ops.get("RequantizeLinear", 0))

    if data_yaml is not None:
        val_imgsz = size[0] if size[0] == size[1] else size
        fn = build_evaluate_fn(model_path, data_yaml, size, device)
        score, extras = fn(graph)
        print("\033[32msimulator mAP50=%.4f  mAP50-95=%.4f\033[0m" % (extras["mAP50"], extras["mAP50-95"]))
    print("\033[32mexported: %s\033[0m" % espdl)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fixed-config scale-aligned int8 quantization (fewer RequantizeLinear).")
    p.add_argument("--model", required=True, help="trained .pt (auto-exported) or .onnx")
    p.add_argument("--size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--target", default="esp32p4")
    p.add_argument("--calib_data", required=True)
    p.add_argument("--espdl", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batchsz", type=int, default=16)
    p.add_argument("--calib_steps", type=int, default=16)
    p.add_argument("--data", default=None, help="dataset yaml to also report simulator mAP")
    a = p.parse_args()
    quantize_aligned(a.model, a.size, a.target, a.calib_data, a.espdl,
                     device=a.device, batchsz=a.batchsz, calib_steps=a.calib_steps, data_yaml=a.data)

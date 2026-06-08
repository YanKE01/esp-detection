import os
import argparse
from pathlib import Path
import shutil
import re

from train import Train
from deploy.quantize_aligned import quantize_aligned


def rename_project(root_dir: Path, replacements: dict):
    target_extensions = {".cpp", ".hpp", ".txt", ".yml"}
    for file in root_dir.rglob("*"):
        if file.suffix in target_extensions or file.name == "Kconfig":
            try:
                content = file.read_text(encoding="utf-8")
                original_content = content
                # find add_custom_command lines
                skipped_lines = re.findall(r".*add_custom_command.*", content)
                placeholders = {}
                for i, line in enumerate(skipped_lines):
                    placeholder = f"__PLACEHOLDER_{i}__"
                    content = content.replace(line, placeholder)
                    placeholders[placeholder] = line
                for old, new in replacements.items():
                    content = content.replace(old, new)
                for placeholder, line in placeholders.items():
                    content = content.replace(placeholder, line)
                if content != original_content:
                    file.write_text(content, encoding="utf-8")
                    print(f"Replaced in: {file}")

            except Exception as e:
                print(f"Failed to process {file}: {e}")


def run(class_name, pretrained_path, dataset, size, target, calib_data, espdl, img, esp_dl_path="../esp-dl"):
    """
    The whole process of realizing a customized detection model, including train, export, quantize a model and deploy it on ESP32 AI chips.
    """
    assert isinstance(size, list) and len(size) == 2, "size should be a list, with len(size) = 2."
    h, w = size
    if h != w:
        print("\033[32mAdopt rect=True training strategy \033[0m")
        print("\033[32mOptional: pre-training \033[0m")
        # results = Train(pretrained_path=pretrained_path, dataset=dataset, imgsz=max(h, w), rect=False)
        print("\033[32mStart training \033[0m")
        # model_path = os.path.join(str(results.save_dir), "weights/best.pt") # use pre-training weights to fine-tune your model
        # results = Train(model_path, dataset, size, epochs=30, rect=True) # fine-tune epochs = 30~50
        results = Train(pretrained_path, dataset, size, rect=True)
    else:
        print("\033[32mStart training \033[0m")
        results = Train(pretrained_path, dataset, size)
    # get the save path of best.pt
    model_path = os.path.join(str(results.save_dir), "weights/best.pt")
    print("\033[32mQuantize model to ESP-DL format (scale-aligned, chip-faithful) \033[0m")
    # quantize_aligned exports .pt -> ONNX internally, then does a single fixed-config,
    # scale-aligned int8 quantization (force_alignment_overlap) so the .espdl matches
    # the on-chip behaviour (avoids the RequantizeLinear sim<->chip mismatch).
    quantize_aligned(
        model_path=model_path,
        size=size,
        target=target,
        calib_data=calib_data,
        espdl=espdl,
    )
    print("\033[32mGenerate CPP Project running on chips \033[0m")
    # Export the project from templates into your EXISTING local esp-dl tree.
    # We never download esp-dl; the generated example/model components use relative
    # override_paths (../../esp-dl, ../../../models/<class>_detect) so they must live
    # inside the esp-dl tree (examples/ + models/) to build.
    assert os.path.isdir(esp_dl_path), (
        f"esp-dl not found at '{esp_dl_path}'. Pass --esp_dl_path pointing to your "
        f"local esp-dl checkout (e.g. --esp_dl_path /path/to/esp-dl)."
    )

    examples_path = os.path.join(esp_dl_path, "examples")
    models_path = os.path.join(esp_dl_path, "models")
    custom_example_path = os.path.join(examples_path, class_name + "_detect")
    custom_model_path = os.path.join(models_path, class_name + "_detect")
    # create folder both in examples and models
    os.makedirs(custom_example_path, exist_ok=True)
    os.makedirs(custom_model_path, exist_ok=True)
    # copy files from template to custom path
    shutil.copytree("deploy/espdet_model_template", custom_model_path, dirs_exist_ok=True)
    shutil.copytree("deploy/espdet_example_template", custom_example_path, dirs_exist_ok=True)

    # Only the file name is embedded (EMBED_FILES / the _binary_<name>_jpg symbol),
    # so use the basename even if --img is given as a full path.
    img_name = os.path.basename(img)
    replacements = {
        "custom": class_name,
        "CUSTOM": class_name.upper(),
        "imgH": str(h),
        "imgW": str(w),
        "espdet.jpg": img_name,
        "espdet_jpg": os.path.splitext(img_name)[0] + "_jpg",
    }

    rename_project(Path(custom_example_path), replacements)
    rename_project(Path(custom_model_path), replacements)

    espdl_model_path = os.path.join(custom_model_path, "models/p4") if target == "esp32p4" else os.path.join(custom_model_path, "models/s3")
    shutil.copy(espdl, espdl_model_path)

    shutil.copy(img, os.path.join(custom_example_path, "main"))

    print("\033[32m"
          "You can run models on chips right now!\n",
          "Please run:\n",
          "cd {custom_example_path}\n",
          "step1: idf.py set-target esp32p4/esp32s3\n",
          "step2: idf.py flash monitor\n"
          "\033[0m")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a custom detection model and deploy it on ESP32-series chips.")
    parser.add_argument("--class_name", type=str, required=True, help="Input object detection target class")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Input pretrained .pt model path")
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset path for train/validate/test")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Input resolution in [h, w] format, e.g. --size 128 224")
    parser.add_argument("--target", type=str, default="esp32p4", help="Input ESP32 chips, e.g. 'esp32p4', 'esp32s3'")
    parser.add_argument("--calib_data", type=str, required=True, help="Input calibration dataset path")
    parser.add_argument("--espdl", type=str, required=True, help="Output ESP-DL model path")
    parser.add_argument("--img", type=str, required=True, help="Input test img path for running on ESP32-chips")
    parser.add_argument("--esp_dl_path", type=str, default="../esp-dl",
                        help="Path to your EXISTING local esp-dl checkout; the CPP project is "
                             "exported into its examples/ and models/ (never downloaded).")

    args = parser.parse_args()
    run(args.class_name, args.pretrained_path, args.dataset, args.size, args.target,
        args.calib_data, args.espdl, args.img, args.esp_dl_path)
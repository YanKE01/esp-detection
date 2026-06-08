import torch
from ultralytics import YOLO
from nn.esp_tasks import custom_parse_model
import ultralytics.nn.tasks as tasks

def Train(pretrained_path=None, dataset="cfg/datasets/coco_cat.yaml", imgsz=224, **kwargs):
    """
    Train espdet_pico on customized dataset.
    :param pretrained_path: the path of pretrained .pt file, default is None.
    :return:
    """
    tasks.parse_model = custom_parse_model  # add ESP-customized block
    # load the model
    if pretrained_path not in [None, 'None']: # use pretrained weights
        model = YOLO(pretrained_path)
    else:
        model = YOLO('cfg/models/espdet_pico.yaml') # # build a new model from YAML if you don't need to load a pretrained model
    train_setting = dict( # you can set your own train settings here.
        data=dataset,
        epochs=1200, # set to a reasonable epoch
        imgsz=imgsz, # input img shape, 224 means input is 224*224. if you want to train with w ≠ h, you need to set rect=True and imgsz=[h, w]
        # espdet_pico is tiny (~0.36M params) so it uses little GPU memory regardless;
        # the bottleneck is data loading, not GPU. Speed it up with cache + workers,
        # not by inflating batch to "fill" the card (huge batch on small datasets =
        # too few steps/epoch and worse convergence). For huge datasets you can set
        # batch to an int, or batch=-1 / batch=0.8 to auto-size to GPU memory.
        batch=256,
        cache=True,        # cache images in RAM (fast IO). Use 'disk' if RAM is tight.
        workers=16,        # dataloader processes; raise if CPU has spare cores
        device=("0" if torch.cuda.device_count() > 0 else "cpu"),  # auto: GPU 0 if available, else CPU
        optimizer='auto',
        close_mosaic=30,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        rect=False,
    )
    train_setting.update(kwargs)
    results = model.train(**train_setting)

    return results

if __name__ == '__main__':
    Train()


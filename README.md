# SyNet
Object Detection with Ensemble Networks

## Summary
A combination method, SyNet, is proposed in this work which combines multi-stage detectors with single-stage ones with the motivation of decreasing the high false negative rate of multi-stage detectors and increasing the quality of the single-stage detector proposals. As building blocks, CenterNet and Cascade R-CNN with pretrained feature extractors are utilized along with an ensembling method.

## Dependencies

These must be installed before next steps.

+ Python 3.5+
+ Tensorflow >= 2.0
+ Tensorpack 0.10
+ Torch 1.1.0
+ Torchvision 0.3.0 


## Training
### 3.1. Include the Dataset

First of all, transfer your dataset in MS-COCO format to tensorpack/examples/FasterRCNN/DATA/ and CenterNet/data/DATA/. Format must be same as:

```text
DATA
├─  annotations
│	├─  instances_train2017.json
│	└─  instances_val2017.json
├─  train2017
│	├─  0.jpg
│	├─  1.jpg
│	└─  ...
└─  val2017
    ├─  0.jpg
    ├─  5.jpg
    └─  ...
```

### 3.2. Training Cascade R-CNN

From http://models.tensorpack.com/#FasterRCNN, download ImageNet-R101-AlignPadding.npz under Faster R-CNN to tendorpack/examples/FasterRCNN/back/ folder. Then, start training by

```
python train.py --config BACKBONE.WEIGHTS= /path/to/tensorpack/examples/FasterRCNN/back/ImageNet-R101-AlignPadding.npz DATA.BASEDIR=/path/to/tensorpack/examples/FasterRCNN/DATA FPN.CASCADE=True FPN.NORM=GN BACKBONE.NORM=GN FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3] TEST.RESULT_SCORE_THRESH=1e-4 PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800] TRAIN.LR_SCHEDULE=9x TRAIN.NUM_GPUS=1 MODE_MASK=False
```
Different training configurations can be found at https://github.com/tensorpack/tensorpack

### 3.3. Training CenterNet

Change the CenterNet/src/lib/datasets after adding your dataset describer to CenterNet/src/lib/datasets/custom_data.py which should be similar CenterNet/src/lib/datasets/dataset/coco.py. Furthermore, modify CenterNet/src/lib/utils/debugger.py and change class names to class names list of yout own dataset. Then, start training by

```
python /path/to/CenterNet/src/main.py ctdet --exp_id coco_dla_2x --dataset custom_data --batch_size 8 --master_batch 32 --lr 5e-4 --gpus 0 --num_workers 0 
```
Different training configurations can be found at https://github.com/xingyizhou/CenterNet.

## Results
### 4.1. COCO Dataset

Results for COCO val-2017 set are presented below with results of the state-of-the-art-methods.

 | Method                       | Feature Extractor | mAP <br/> (0.05:0.95) | mAP <br/> (0.50) | mAP <br/> (0.75)                                                                                                                                                                                                                                                                                                                                   |
 | -                              | -                                                                       | :-:                                                  | :-:                      | :-:                                                                                                                                                                                                                                                                                                                                                                        |
 | SyNet (ours)                         | ResNet101 + DLA-34                                                                   | 50.1                                                | **68.9**             |**54.3**                                                                                                                                                                                                           |
 | Cascade R-CNN                        | ResNet101                                                                    | 42.7                                                | 61.6                  | 46.6                                                                                                                                                                                                                                                                                          |
 | Cascade R-CNN                        | ResNet50                                                                    | 40.3                                                | 59.4                  | 43.7                                                                                                                                                                                                                                                                                                         |
 | CenterNet                               | ResNet50                                                                    | 40.3                                                  | 59.1                    | 44.0                                                                                                                                                                                                                                                                                                         |
 | CenterNet                               | ResNet50                                                                    | 37.4                                                  | 55.1                    | 40.8                                                                                                                                                                                                                                                                                                         |
 | Faster R-CNN                               | ResNet50                                                                    | 38.5                                                  | 60.3                    | 41.6                                                                                                                                                                                                                                                                                                         |
 | Faster R-CNN                               | ResNet50                                                                    | 36.4                                                  | 58.4                    | 39.1                                                                                                                                                                                                                                                                                                         |
 | Mask R-CNN                               | ResNet50                                                                    | 39.4                                                  | 60.9                    | 43.3                                                                                                                                                                                                                                                                                                         |
 | Mask R-CNN                               | ResNet50                                                                    | 37.3                                                  | 59.0                    | 40.2                                                                                                                                                                                                                                                                                                         |
 | Retina Net                               | ResNet50                                                                    | 37.7                                                  | 57.5                    | 40.4                                                                                                                                                                                                                                                                                                         |
 | Retina Net                              | ResNet50                                                                    | 35.6                                                  | 55.5                    | 38.3                                                                                                                                                                                                                                                                                                         |
 | Cascade Mask R-CNN                              | ResNet50                                                                    | 42.6                                                  | 60.7                    | 46.7                                                                                                                                                                                                                                                                                                         |
 | Cascade Mask R-CNN                              | ResNet50                                                                    | 41.2                                                  | 59.1                    | 45.1                                                                                                                                                                                                                                                                                                         |
 | Hybrid Task Cascade                              | ResNet50                                                                    | 44.9                                                  | 63.8                    | 48.7                                                                                                                                                                                                                                                                                                         |
 | Hybrid Task Cascade                              | ResNet50                                                                    | 43.2                                                  | 62.1                    | 46.8                                                                                                                                                                                                                                                                                                         |
 | EfficientDet-D7 (1536)                              | ResNet50                                                                    | **52.1**                                                  | -                    | -                                                                                                                                                                                                                                                                                                         |

### 4.1. VisDrone Dataset

Results for VisDrone test-set are presented below with results of the state-of-the-art-methods.

| Method                        | mAP <br/> (0.05:0.95) | mAP <br/> (0.50) | mAP <br/> (0.75)                                                                                                                                                                                                                                                                                                                                   |
 | -                                                                                                    | :-:                                                  | :-:                      | :-:                                                                                                                                                                                                                                                                                                                                                                        |
 | SyNet (ours)                                                                                            | **29.21**                                                | 52.42             |**28.56**                                                                                                                                                                                                           |
 | RRNet                                                                                            | 29.13                                                | **55.82**                  | 27.23                                                                                                                                                                                                                                                                                          |
 | CornerNet                                                                                           | 17.41                                                | 34.12                  | 15.78                                                                                                                                                                                                                                                                                                         |
 | Light R-CNN                                                                                                   | 16.53                                                  | 32.78                    | 15.13                                                                                                                                                                                                                                                                                                         |
 | FPN                                                                                                   | 16.51                                                  | 32.20                    | 14.91                                                                                                                                                                                                                                                                                                         |
 | Cascade R-CNN                                                                                                   | 16.09                                                  | 31.91                    | 15.01                                                                                                                                                                                                                                                                                                         |
 | Detnet59                                                                                                   | 15.26                                                  | 29.23                    | 14.34                                                                                                                                                                                                                                                                                                         |
 | RefineDet                                                                                                   | 14.9                                                  | 28.76                    | 14.08                                                                                                                                                                                                                                                                                                         |
 | Retina Net                                                                                                  | 11.81                                                  | 21.37                    | 11.62                                                                                                                                                                                                                                                                                                         |

## Acknowledgement
For the training of the Cascade R-CNN, Tensorpack is used: https://github.com/tensorpack/tensorpack
For the training of the CenterNet, CenterNet is used: https://github.com/xingyizhou/CenterNet
For the weighted box ensemble,  https://github.com/ZFTurbo/Weighted-Boxes-Fusion is used.

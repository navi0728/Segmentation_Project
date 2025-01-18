# Segmentation Task

## Toy Project Topic
- Human Segmentation using U-Net model with Kaggle image datasets

- Model: U-Net: Convolutional Networks for Biomedical Image Segmentation [paper](https://arxiv.org/abs/1505.04597) 

## 1. Download Dataset 
- [Sketch Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset/data)

## 2. Training And Validation
```
python train.py
```

## Results  

### Prediction
<img src=https://github.com/navi0728/Segmentation_Project/blob/main/unet_tiktok/assets/output1.png width="750" height="300"/>

<img src=https://github.com/navi0728/Segmentation_Project/blob/main/unet_tiktok/assets/output2.png width="750" height="300"/>

### Training Loss Results
<img src=https://github.com/navi0728/Segmentation_Project/blob/main/unet_tiktok/assets/training_loss_trend.png width="700" height="300"/>

### Best IoU Results
<img src=https://github.com/navi0728/Segmentation_Project/blob/main/unet_tiktok/assets/validation_iou_trend.png width="700" height="300"/>



## Structure
```
.
├── date #Download dataset
│   ├── images 
│   └── masks
├── new_results #Resized masks
├── results #Predicted masks
├── dataset.py 
├── model.py
├── new_results.py
├── show.py
└── train.py
```

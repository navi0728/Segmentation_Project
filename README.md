# Segmentation Task

## Toy Project 
- **Topic**: Human Segmentation using U-Net model with Kaggle image datasets
- **Model**: U-Net: Convolutional Networks for Biomedical Image Segmentation [paper](https://arxiv.org/abs/1505.04597) 

### Demo Video
![Demo GIF](https://github.com/navi0728/Segmentation_Project/blob/main/unet_tiktok/assets/Demo.gif)

## 1. Download Dataset 
- [Sketch Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset/data)
- **EDA**  
  - **Images**: Extracted from video frames  
    - **Number**: 2615  
    - **Shape**: (3, 960, 540)  
  - **Masks**: Segmented images of dancing people  
    - **Number**: 2615  
    - **Shape**: (1, 960, 540)  
  - **Features**:  
    - No annotation file. Only masked image files.  
    - Not split into train, validation, and test datasets.

## 2. Training And Validation
```
python train.py
```

## 3. Inference
```
python infer.py
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
unet_tiktok
│
├── assets
├── date #Download dataset
│   ├── images 
│   └── masks
├── networks
|   └── model.py
├── new_results #Resize Predicted masks
├── results #Predicted masks
├── saves
|    └── 0000
|    └──...
├── utills
├── infer.py
├── show.py
├── test_img.png
└── train.py
```

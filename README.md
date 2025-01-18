# Segmentation Task

## Toy Project Topic
Human Segmentation using U-Net model with Kaggle image datasets

- Model: U-Net: Convolutional Networks for Biomedical Image Segmentation [paper](https://arxiv.org/abs/1505.04597) 

## 1. Download Dataset 
- [Sketch Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset/data)

## 2. Trainning And Validing

```
python train.py
```

## Results   
### Sketch Image
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Sketch_Image.png width="200" height="200"/>

### Step 1: Sketch2Image  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step1_output.png width="200" height="200"/>

### Step 2: Image2Background  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step2_output.png width="200" height="200"/>

### Step 3: Background2Movement  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step3_output.gif width="200" height="200"/>



## structure
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

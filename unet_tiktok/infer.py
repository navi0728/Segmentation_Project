from PIL import Image
#from train import transform
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import os
import json
import torch
import cv2
import numpy as np

save_root_folder = "./test_results"
# test할 dataset load
img_path = r"C:\Users\minju\Desktop\code\unet_tiktok\Segmentation_Project\test_img.png"

# 기존 img.shape -> torch.Size([4, 28, 28])
img = Image.open(img_path).convert("RGB")

# Data preprocessing
transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

img = transform(img)

# model load
class encoder_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    

class decoder_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
    
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class upsample_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.upconv(x)
        return x

class bottleneck_block(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = encoder_block(3, 64)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.enc4 = encoder_block(256, 512)
        self.bottle = bottleneck_block(512, 1024, 1024)

        self.dec1 = decoder_block(1024, 512) 
        self.dec2 = decoder_block(512, 256)
        self.dec3 = decoder_block(256, 128)
        self.dec4 = decoder_block(128, 64)

        self.up1 = upsample_block(1024, 512)
        self.up2 = upsample_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.up4 = upsample_block(128, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        # self.initialize_weights()
            

    def forward(self, x):
        # encoder
        x = self.enc1(x)
        x1 = x
        x = F.max_pool2d(x, 2)
        x = self.enc2(x)
        x2 = x
        x = F.max_pool2d(x, 2)
        x = self.enc3(x)
        x3 = x
        x = F.max_pool2d(x, 2)
        x = self.enc4(x)
        x4 = x
        x = F.max_pool2d(x, 2)
        x = self.bottle(x)
        
        # decoder
        x = self.up1(x)
        x = self.dec1(x, x4)
        x = self.up2(x)
        x = self.dec2(x, x3)
        x = self.up3(x)
        x = self.dec3(x, x2)
        x = self.up4(x)
        x = self.dec4(x ,x1)
        
        # head
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

# train.py에서 생성한 하위 폴더 안의 weight와 json 파일 가져오기   (ex saves\0001)
trained_folder = r"C:\Users\minju\Desktop\code\unet_tiktok\Segmentation_Project\unet_tiktok\saves\0002"

hyparam_path = os.path.join(trained_folder,"hyper.json")

with open(hyparam_path, "r") as f:
    hyparam_dict = json.load(f)

# 모델 껍데기 생성 
model = Unet(1)

trained_model_path = os.path.join(trained_folder, "best.pt")

# torch.save <-> torch.load
trained_model = torch.load(trained_model_path, weights_only=False)

# model.state_dict()로 저장한 weight를 모델 껍데기에 넣기기
model.load_state_dict(trained_model)

#학습된 이미지 shape과 동일해야함함
img = img.unsqueeze(0)
prediction = model(img)
prediction[prediction >= 0.5] = 1
prediction[prediction < 0.5] = 0


img_name = (img_path.split('\\'))[-1]

result_mask = prediction.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
result_mask = (result_mask * 255.0).astype(np.uint8)

result_img_path = os.path.join(trained_folder, img_name)
cv2.imwrite(result_img_path, result_mask)


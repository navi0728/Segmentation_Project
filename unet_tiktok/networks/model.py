import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.ones_(m.weight)
    #             nn.init.zeros_(m.bias)


# if __name__ == "__main__":
#     test_input = torch.randn(5, 3, 256, 256)
#     model = Unet(1, enc_dilation=[1, 2, 2, 4])
#     output = model(test_input)
#     print(output.shape)

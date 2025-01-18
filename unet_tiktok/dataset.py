import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from glob import glob

# Custom Dataset
class Dancing_Dataset(Dataset):
    def __init__(self, root_path):
        # self.imgs_list = []
        # self.masks_list = []

        # imgs_path = os.path.join(root_path, "images").replace("\\", "/")
        # masks_path = os.path.join(root_path, "masks").replace("\\", "/")

        # for (path, _, files) in os.walk(imgs_path):
        #     for filename in files:
        #         ext = os.path.splitext(filename)[-1]
        #         if ext == '.png':
        #             target_img_path = os.path.join(path, filename)
        #             target_mask_path = target_img_path.replace("images", "masks")
        #             self.imgs_list.append(target_img_path)
        #             self.masks_list.append(target_mask_path)

        self.imgs_path = sorted(glob(f'{root_path}/images/*.png'))
        self.masks_path = sorted(glob(f'{root_path}/masks/*.png'))

        # Data Processing
        self.image_transform = transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_transform = transforms.Compose([
                            transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
                            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img_path = img_path.replace("\\", "/")
        mask_path = self.masks_path[idx]
        mask_path = mask_path.replace("\\", "/")
        img_name = (img_path.split('/'))[-1]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        tensor_img = self.image_transform(img)
        tensor_mask = self.mask_transform(mask)
        
        return tensor_img, tensor_mask, img_name


#if __name__ == "__main__":
     #my_dataset = Dancing_Dataset(root_path="C:\\Users\\minju\Desktop\\code\\unet_tiktok\data")
     #img, mask,img_name = my_dataset.__getitem__(1)
     #print(img.shape, mask.shape) -> torch.Size([3, 512, 512]) torch.Size([1, 512, 512])
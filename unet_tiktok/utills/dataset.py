from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from PIL import Image
from glob import glob

# Custom Dataset
class Dancing_Dataset(Dataset):
    def __init__(self, args):
        
        self.imgs_path = sorted(glob(f'{args.data_folder}\images\*.png'))
        self.masks_path = sorted(glob(f'{args.data_folder}\masks\*.png'))
        # print(f"Found {len(self.imgs_path)} image files.")

        # Data Processing
        self.image_transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_transform = transforms.Compose([
                            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
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

# data split
def train_val_dataset(dataset, val_split=0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets
    
def get_dataloader(args, datasets):
    # Dataloader
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(datasets['val'], batch_size=args.batch_size)

    return train_loader, test_loader


# if __name__ == "__main__":
     #my_dataset = Dancing_Dataset(root_path="C:\\Users\\minju\Desktop\\code\\unet_tiktok\data")
     #img, mask,img_name = my_dataset.__getitem__(1)
     #print(img.shape, mask.shape) -> torch.Size([3, 512, 512]) torch.Size([1, 512, 512])
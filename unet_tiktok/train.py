from dataset import Dancing_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from model import Unet
from torch.utils.data import Subset
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import os
import cv2


batch_size = 4
epoch = 25
learning_rate = 0.001
save_root_folder = "./results2"

loss_history = [] 
iou_history = []

my_dataset = Dancing_Dataset(root_path="./data")

# data split
def train_val_dataset(dataset, val_split=0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

datasets = train_val_dataset(my_dataset)

# print(len(datasets['train'])) -> 2092
# print(len(datasets['val'])) -> 523

train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets['val'], batch_size=batch_size)


model = Unet(1,enc_dilation=[1, 2, 2, 4]).cuda()

optimizer = Adam(params=model.parameters(), lr=learning_rate)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.BCELoss()

## Training
best_iou = 0

for i in range(epoch):
    for j, (tensor_img, tensor_mask, img_name) in enumerate(train_loader):
        model.train()
        tensor_img, tensor_mask = tensor_img.cuda(), tensor_mask.cuda()
        #print(tensor_img.shape)
        #print(tensor_mask.shape)
        output = model(tensor_img)
        loss = loss_fn(output, tensor_mask)
        loss.backward()
        optimizer.step()  
        optimizer.zero_grad() 

        loss_history.append(loss.item())

        ## Validation
        if j % 100 == 0:
            print(f'epoch : {i}/{epoch} | step : {j} | loss value : {loss}')
            
            with torch.no_grad():
                model.eval()

                correct = 0
                total = 0

                data_len = 0
                iou_scores = []

                if not os.path.exists(save_root_folder):
                    os.makedirs(save_root_folder, exist_ok=True)

                for v_image, v_label, v_img_name in test_loader:
                    v_image, v_label = v_image.cuda(), v_label.cuda()
                    v_predic = model(v_image)

                    # save_image
                    B = v_predic.shape[0]
                    for b in range(B):
                        target_image_name = v_img_name[b]
                        result_mask = v_predic[b, :, :, :]
                        result_mask = result_mask.cpu().detach()
                        # result_mask : [1, 256, 256]
                        
                        result_mask = result_mask.permute(1, 2, 0)
                        np_result_mask = result_mask.numpy()
                        np_result_mask = (np_result_mask * 255.0).astype(np.uint8)

                        result_img_path = os.path.join(save_root_folder, target_image_name)
                        cv2.imwrite(result_img_path, np_result_mask)
                        
                    # v_predic : [B, 1, 256, 256]
                    # v_label : [B, 1, 256, 256]

                    v_predic[v_predic >= 0.5] = 1
                    v_predic[v_predic < 0.5] = 0

                    intersection = torch.sum(v_predic * v_label)
                    union = torch.sum(v_predic + v_label) - intersection
                    iou = intersection / (union + 1e-6)
                    
                    iou_scores.append(iou.item())

                avg_iou = np.mean(iou_scores)
                iou_history.append(avg_iou)

                print(f"Validation IoU: {avg_iou}")

                if avg_iou > best_iou:
                    best_iou = avg_iou
        
                model.train()

            print(f"Best IoU: {best_iou}")
    
    lr_scheduler.step()

# visualize loss trend
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss Value')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

save_path = os.path.join(save_root_folder, "training_loss.png")
plt.savefig(save_path)
print(f"Loss trend graph saved at: {save_path}")

# visualize IoU trend
plt.figure(figsize=(10, 5))
plt.plot(iou_history, label='Validation IoU', color='orange')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.title('Validation IoU')
plt.legend()
plt.grid(True)

iou_graph_path = os.path.join(save_root_folder, "validation_iou_trend.png")
plt.savefig(iou_graph_path)
print(f"IoU trend graph saved at: {iou_graph_path}")

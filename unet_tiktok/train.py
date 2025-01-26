from utills.parse import parse_args
from utills.dataset import Dancing_Dataset
from utills.tools import get_params_json, get_result_folder
from utills.dataset import train_val_dataset
from utills.dataset import get_dataloader
from utills.eval import val_segmentation
from torch.optim.lr_scheduler import StepLR
from networks.model import Unet
from torch.optim import Adam
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os


def main():
    args = parse_args()

    result_folder = get_result_folder(args)
    get_params_json(result_folder, args)

    my_dataset = Dancing_Dataset(args)

    datasets = train_val_dataset(my_dataset)

    # print(len(datasets['train'])) -> 2092
    # print(len(datasets['val'])) -> 523

    train_loader, val_loader = get_dataloader(args, datasets)

    model = Unet(1).cuda()

    optimizer = Adam(params=model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.BCELoss()

    ## Training
    best_iou = 0

    loss_history = [] 
    iou_history = []

    for i in range(args.epoch):
        for j, (tensor_img, tensor_mask, _) in enumerate(train_loader):
            tensor_img, tensor_mask = tensor_img.cuda(), tensor_mask.cuda()
            #print(tensor_img.shape)
            #print(tensor_mask.shape)
            optimizer.zero_grad()
            output = model(tensor_img)
            loss = loss_fn(output, tensor_mask)
            loss.backward()
            optimizer.step()   

            loss_history.append(loss.item())

            ## Validation
            if j % 100 == 0:
                print(f'epoch : {i}/{args.epoch} | step : {j} | loss value : {loss}')
                                
                avg_iou = val_segmentation(args, model, val_loader)
                iou_history.append(avg_iou)

                if avg_iou > best_iou:
                    print(f"모델 weight 저장. 이전 acc {best_iou*100:.2f}%, 갱신 acc {avg_iou*100:.2f}%")
                    best_iou = avg_iou
                    torch.save(model.state_dict(), os.path.join(result_folder,"best.pt"))

                model.train()

                print(f"Best IoU: {best_iou}")

        lr_scheduler.step()

if __name__ == '__main__':
    main()
# # visualize loss trend
# plt.figure(figsize=(10, 5))
# plt.plot(loss_history, label='Training Loss')
# plt.xlabel('Steps')
# plt.ylabel('Loss Value')
# plt.title('Training Loss')
# plt.legend()
# plt.grid(True)

# save_path = os.path.join(save_root_folder, "training_loss.png")
# plt.savefig(save_path)
# print(f"Loss trend graph saved at: {save_path}")

# # visualize IoU trend
# plt.figure(figsize=(10, 5))
# plt.plot(iou_history, label='Validation IoU', color='orange')
# plt.xlabel('Epochs')
# plt.ylabel('IoU Score')
# plt.title('Validation IoU')
# plt.legend()
# plt.grid(True)

# iou_graph_path = os.path.join(save_root_folder, "validation_iou_trend.png")
# plt.savefig(iou_graph_path)
# print(f"IoU trend graph saved at: {iou_graph_path}")

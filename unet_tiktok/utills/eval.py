import torch
import os
import numpy as np
import cv2


def val_segmentation(args, model, val_loader):    
    with torch.no_grad():
        model.eval()
        iou_scores = []

        if not os.path.exists(args.save_root_folder):
            os.makedirs(args.save_root_folder, exist_ok=True)

        for v_image, v_label, v_img_name in val_loader:
            v_image, v_label = v_image.cuda(), v_label.cuda()
            v_predic = model(v_image)

            v_predic[v_predic >= 0.5] = 1
            v_predic[v_predic < 0.5] = 0

            intersection = torch.sum(v_predic * v_label)
            union = torch.sum(v_predic + v_label) - intersection
            iou = intersection / (union + 1e-8)
            
            iou_scores.append(iou.item())

            # save_image
            B = v_predic.shape[0]
            for b in range(B):
                target_image_name = v_img_name[b]
                
                # result_mask : [1, 256, 256]
                
                result_mask = v_predic[b].cpu().numpy().transpose(1, 2, 0)
                result_mask = (result_mask * 255.0).astype(np.uint8)

                result_img_path = os.path.join(args.save_root_folder, target_image_name)
                result_img_path = result_img_path.replace("/", "\\")
                cv2.imwrite(result_img_path, result_mask)
                
            # v_predic : [B, 1, 256, 256]
            # v_label : [B, 1, 256, 256]

        avg_iou = np.mean(iou_scores)

    return avg_iou
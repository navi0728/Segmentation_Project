import os
import cv2

save_root_folder = "./results"

new_results_path = "./new_results"
os.makedirs(new_results_path, exist_ok=True)

# 저장된 이미지 리사이즈
for filename in os.listdir(save_root_folder):
    img_path = os.path.join(save_root_folder, filename)
    
    # 이미지 로드 시도
    img = cv2.imread(img_path)
    
    if img is not None:
        # 이미지가 로드되었다면 리사이즈 후 저장
        img = cv2.resize(img, (540, 960))
        cv2.imwrite(os.path.join(new_results_path, filename), img)  
    else:
        pass
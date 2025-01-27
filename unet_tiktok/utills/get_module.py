from torchvision import transforms

def load_transform():

    # Data preprocessing
    transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    return transform

def data_preprocessing(img):
    transform = load_transform()
    img = transform(img)

    return img

def load_model():
    from networks.model import Unet
    model = Unet(1)

    return model

def data_post_processing(new_results_path, args):
    import os
    import cv2

    if new_results_path == "./new_results":

        if not os.path.exists(new_results_path):
            os.makedirs(new_results_path, exist_ok=True)

        # 저장된 이미지 리사이즈
        for filename in os.listdir(args.save_root_folder):
            img_path = os.path.join(args.save_root_folder, filename)
            
            # 이미지 로드 시도
            img = cv2.imread(img_path)
            
            if img is not None:
                # 이미지가 로드되었다면 리사이즈 후 저장
                img = cv2.resize(img, (540, 960))
                cv2.imwrite(os.path.join(new_results_path, filename), img)  
            else:
                pass

    else:
        # 이미지 로드 시도
        img = cv2.imread(new_results_path)
        
        if img is not None:
            # 이미지가 로드되었다면 리사이즈 후 저장
            img = cv2.resize(img, (540, 960))
            filename = (new_results_path.split('\\'))[-1]
            cv2.imwrite(os.path.join(args.trained_folder, "resized_" + filename), img)  
        else:
            pass
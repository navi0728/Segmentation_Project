import os
import json

def get_result_folder(args):
    ## 학습된 모델 weight와 해당 학습으로 사용된 hyperparms를 저장하기 위한 폴더 생성과정
    # 상위 root folder 생성
    if not os.path.exists(args.result_root_folder):
        os.makedirs(args.result_root_folder, exist_ok=True)

        # 하위 folder 생성
    if len(os.listdir(args.result_root_folder))==0:
        name = "0000"

    else:
        sorted(os.listdir(args.result_root_folder))
        number = int(sorted(os.listdir(args.result_root_folder))[-1]) + 1
        name = str(number).zfill(4)

    result_folder = os.path.join(args.result_root_folder, name)
    os.makedirs(result_folder)

    return result_folder

def get_params_json(result_folder, args):
    with open(os.path.join(result_folder,"hyper.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

def load_img(ingfer_args):
    from PIL import Image
    img = Image.open(ingfer_args.img_path).convert("RGB")

    return img

def load_params(infer_args):
    # train.py에서 생성한 하위 폴더 안의 weight와 json 파일 가져오기   (ex saves\0001)
    hyparam_path = os.path.join(infer_args.trained_folder ,"hyper.json")

    with open(hyparam_path, "r") as f:
        hyparam_dict = json.load(f)

    trained_model_path = os.path.join(infer_args.trained_folder , "best.pt")

    return hyparam_dict, trained_model_path

def write_result_mask(infer_args, result_mask):
    import cv2
    img_name = (infer_args.img_path.split('\\'))[-1]

    result_img_path = os.path.join(infer_args.trained_folder, img_name)
    cv2.imwrite(result_img_path, result_mask)

    return result_img_path


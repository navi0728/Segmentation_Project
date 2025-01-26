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
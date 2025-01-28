# import torch
# import numpy as np

# from utills.parse import parse_infer_args
# from utills.tools import load_img, load_params, write_result_mask
# from utills.get_module import data_preprocessing, load_model, data_post_processing

# def main():

#     infer_args = parse_infer_args()

#     img = load_img(infer_args)
#     img = data_preprocessing(img)

#     _, trained_model_path = load_params(infer_args)
#     # torch.save <-> torch.load
#     trained_model = torch.load(trained_model_path, weights_only=False)
#     model = load_model()
#     # model.state_dict()로 저장한 weight를 모델 껍데기에 넣기기
#     model.load_state_dict(trained_model)

#     #학습된 이미지 shape과 동일해야함
#     img = img.unsqueeze(0)

#     prediction = model(img)
#     prediction[prediction >= 0.5] = 1
#     prediction[prediction < 0.5] = 0

#     result_mask = prediction.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     result_mask = (result_mask * 255.0).astype(np.uint8)

#     # mask 이미지 저장
#     new_results_path = write_result_mask(infer_args, result_mask)
    
#     # mask 이미지 resize
#     data_post_processing(new_results_path, infer_args)

# if __name__ == "__main__":
#     main()


import torch
import numpy as np
from PIL import Image

from utills.parse import parse_infer_args
from utills.tools import load_params
from utills.get_module import data_preprocessing, load_model

def main(image):

    infer_args = parse_infer_args()

    img = data_preprocessing(image)

    _, trained_model_path = load_params(infer_args)
    trained_model = torch.load(trained_model_path, weights_only=False)
    model = load_model()
    model.load_state_dict(trained_model)

    img = img.unsqueeze(0)

    prediction = model(img)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0

    result_mask = prediction.detach().cpu().numpy().squeeze(0)  # (C, H, W)
    result_mask = result_mask.squeeze(0)  # (H, W)
    result_mask = (result_mask * 255.0).astype(np.uint8)

    result_image = Image.fromarray(result_mask, mode="L")  # 흑백 이미지 변환
    result_image = result_image.resize((image.size[0], image.size[1]), Image.NEAREST)  # 원본 이미지 크기로 resize
    return result_image


if __name__ == "__main__":
    main()
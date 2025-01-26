import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5, help='데이터 로더가 불러오는 데이터의 수')
    parser.add_argument('--epoch', type=int, default=5, help='학습 횟수')
    parser.add_argument('--result_root_folder', type=str, default='./saves', help='학습된 결과와 hparam를 저장하는 최상위 폴더 이름')
    #parser.add_argument('--resize', type=int, default=256, help='resize할 이미지 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--save_root_folder', type=str, default='./results', help='예측 mask 이미지 저장 폴더 이름')
    parser.add_argument('--data_folder', type=str, default='./data', help='데이터셋 저장 폴더 이름름')

    args = parser.parse_args()

    return args
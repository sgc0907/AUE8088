import argparse
import yaml
import os
from pathlib import Path
import numpy as np

# 필요한 클래스와 함수들을 임포트합니다.
from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.autoanchor import kmean_anchors
from utils.general import LOGGER, colorstr

def main(opt):
    """
    KAIST 데이터셋에 맞는 전용 로더와 모든 필수 인자를 사용하여 최적의 앵커를 계산합니다.
    """
    print("YOLOv5 Anchor Calculation Script (stride 문제 해결 최종본)")
    print("─" * 60)

    # 1. 데이터 및 모델 설정 파일(.yaml)을 로드합니다.
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    with open(opt.model) as f:
        model_dict = yaml.safe_load(f)
        
    n = model_dict['anchors']
    num_anchors = len(n) * len(n[0]) // 2
    
    # --- 핵심 수정 ---
    # stride 값은 yaml에 없으므로, yolov5s의 표준 최대 stride인 32로 직접 설정합니다.
    max_stride = 32
    print(f"'{opt.model}' 파일에서 총 {num_anchors}개의 앵커를 확인했습니다.")
    print(f"모델의 최대 stride는 {max_stride}로 가정하여 진행합니다.")
    # --- 수정 끝 ---

    # 2. 데이터 경로를 조합하고 KAIST 전용 데이터셋 객체를 생성합니다.
    train_path_str = '' 
    try:
        ROOT = Path(os.getcwd())
        datapath = Path(data_dict.get('path') or '')
        if not datapath.is_absolute():
            datapath = (ROOT / datapath).resolve()
        
        train_path_source = data_dict['train']
        train_path_file = train_path_source[0] if isinstance(train_path_source, list) else train_path_source
            
        train_path_str = str(datapath / train_path_file)
        LOGGER.info(f"{colorstr('green', 'Resolved training data path:')} {train_path_str}")

        dataset = LoadRGBTImagesAndLabels(
            path=train_path_str,
            img_size=opt.img_size,
            batch_size=1,
            augment=True,
            hyp=None,
            rect=True,
            image_weights=False,
            cache_images=False,
            single_cls=False,
            stride=int(max_stride),
            pad=0.0
        )

    except Exception as e:
        LOGGER.error(f"데이터 로딩 중 오류 발생: {e}")
        if train_path_str:
            LOGGER.error(f"'{train_path_str}' 파일 경로 및 내용을 확인해주세요.")
        return

    # 3. 생성된 데이터셋 객체를 사용하여 k-means 앵커 계산을 수행합니다.
    anchors = kmean_anchors(dataset=dataset, n=num_anchors, img_size=opt.img_size, thr=4.0, gen=1000, verbose=True)
    anchors = anchors.reshape(len(n), -1)
    
    print("\n✅ 앵커 계산 완료!")
    print("─" * 60)
    print(f"아래의 새로운 앵커 값을 '{opt.model}' 파일의 'anchors:' 섹션에 복사하세요.")
    print("\nanchors:")
    for a in anchors:
        print(f"  - {np.round(a, 2).tolist()}")
    print("─" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/kaist-rgbt.yaml')
    parser.add_argument('--model', type=str, default='models/yolov5s_kaist-rgbt.yaml')
    parser.add_argument('--img-size', type=int, default=640)
    opt = parser.parse_args()
    main(opt)
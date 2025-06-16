import argparse
import yaml
import os
from pathlib import Path
import numpy as np


from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.autoanchor import kmean_anchors
from utils.general import LOGGER, colorstr

def main(opt):
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    with open(opt.model) as f:
        model_dict = yaml.safe_load(f)
        
    n = model_dict['anchors']
    num_anchors = len(n) * len(n[0]) // 2
    max_stride = 32

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
        LOGGER.error(f"Error loading data: {e}")
        if train_path_str:
            LOGGER.error(f"Please check the path and contents of '{train_path_str}'")
        return

    anchors = kmean_anchors(dataset=dataset, n=num_anchors, img_size=opt.img_size, thr=4.0, gen=1000, verbose=True)
    anchors = anchors.reshape(len(n), -1)
    
    print("\nanchors:")
    for a in anchors:
        print(f"  - {np.round(a).astype(int).tolist()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/kaist-rgbt.yaml')
    parser.add_argument('--model', type=str, default='models/yolov5s_kaist-rgbt.yaml')
    parser.add_argument('--img-size', type=int, default=640)
    opt = parser.parse_args()
    main(opt)
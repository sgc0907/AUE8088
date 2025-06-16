#!/bin/bash

# 사용할 GPU를 2번으로 지정
export CUDA_VISIBLE_DEVICES=2

# val.py 실행
python3 val.py \
    --img 640 \
    --batch-size 32 \
    --data data/kaist-rgbt.yaml \
    --weights runs/train/yolov5n-rgbt31/weights/best.pt \
    --workers 16 \
    --name yolov5n-rgbt-val \
    --rgbt \
    --single-cls \
    --task test
#!/bin/bash

# 사용할 GPU를 2번으로 지정
export CUDA_VISIBLE_DEVICES=2

# val.py 실행
python3 val.py \
    --img 640 \
    --batch-size 32 \
    --data data/kaist-rgbt.yaml \
    --weights runs/train/yolov5s-rgbt9/weights/best.pt \
    --workers 0 \
    --name yolov5s-rgbt-val \
    --rgbt \
    --task test \
    --save-json
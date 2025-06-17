CUDA_VISIBLE_DEVICES=3 python3 train_simple.py \
  --img 640 \
  --batch-size 16 \
  --epochs 100 \
  --data data/kaist-rgbt.yaml \
  --cfg models/yolov5s_kaist-rgbt.yaml \
  --weights yolov5s.pt \
  --workers 16 \
  --name yolov5s-rgbt \
  --entity $WANDB_ENTITY \
  --rgbt \
  --hyp data/hyps/hyp.kaist-custom.yaml


# CUDA_VISIBLE_DEVICES=2 python3 train_simple.py \
#   --img 640 \
#   --batch-size 32 \
#   --epochs 100 \
#   --data data/kaist-rgbt.yaml \
#   --cfg models/yolov5n_kaist-rgbt.yaml \
#   --weights yolov5n.pt \
#   --workers 16 \
#   --name yolov5n-rgbt \
#   --entity $WANDB_ENTITY \
#   --rgbt \
#   --hyp data/hyps/hyp.kaist-custom.yaml



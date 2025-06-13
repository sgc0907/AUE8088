import random
import os

# 원본 학습 데이터 파일 경로
# kaist-rgbt.yaml 파일에 따르면 'datasets/kaist-rgbt/' 디렉토리 안에 있습니다.
# 스크립트를 yolov5 디렉토리에서 실행한다고 가정합니다.
original_data_file = 'datasets/kaist-rgbt/train-all-04.txt'

# 새로 생성될 학습 및 검증 데이터 파일 경로
output_dir = os.path.dirname(original_data_file)
train_file = os.path.join(output_dir, 'train.txt')
val_file = os.path.join(output_dir, 'val.txt')

# 검증 데이터셋 비율 (20%)
val_split_ratio = 0.2

# 원본 파일을 읽어들입니다.
try:
    with open(original_data_file, 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"오류: '{original_data_file}' 파일을 찾을 수 없습니다.")
    print("YOLOv5 프로젝트의 최상위 디렉토리에서 스크립트를 실행하고 있는지 확인해주세요.")
    exit()

# 데이터 순서를 섞어서 무작위로 분할되도록 합니다.
random.shuffle(lines)

# 데이터를 나눌 인덱스를 계산합니다.
split_index = int(len(lines) * (1 - val_split_ratio))

# 데이터를 학습 및 검증 세트로 나눕니다.
train_lines = lines[:split_index]
val_lines = lines[split_index:]

# 학습 데이터 파일을 작성합니다.
with open(train_file, 'w') as f:
    f.writelines(train_lines)

# 검증 데이터 파일을 작성합니다.
with open(val_file, 'w') as f:
    f.writelines(val_lines)

print("✅ 데이터 분할이 완료되었습니다!")
print("─" * 30)
print(f"총 데이터 수: {len(lines)}")
print(f"학습 데이터: {len(train_lines)}개 -> {train_file}")
print(f"검증 데이터: {len(val_lines)}개 -> {val_file}")
print("─" * 30)
print("\n이제 'kaist-rgbt.yaml' 파일을 수정해야 합니다.")
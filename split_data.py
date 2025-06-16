import random
import os


original_data_file = 'datasets/kaist-rgbt/train-all-04.txt'

output_dir = os.path.dirname(original_data_file)
train_file = os.path.join(output_dir, 'train.txt')
val_file = os.path.join(output_dir, 'val.txt')

val_split_ratio = 0.2

try:
    with open(original_data_file, 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Error: '{original_data_file}'")
    exit()

random.shuffle(lines)

split_index = int(len(lines) * (1 - val_split_ratio))
train_lines = lines[:split_index]
val_lines = lines[split_index:]


with open(train_file, 'w') as f:
    f.writelines(train_lines)
with open(val_file, 'w') as f:
    f.writelines(val_lines)

print("Data split completed successfully")
print("─" * 30)
print(f"Total data count: {len(lines)}")
print(f"Training data: {len(train_lines)} -> {train_file}")
print(f"Validation data: {len(val_lines)} -> {val_file}")
print("─" * 30)
print("\nPlease update the 'kaist-rgbt.yaml' file accordingly")
import os
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

def process_class_dir(class_dir_path, pbar=False):
  records = []
  class_name = os.path.basename(class_dir_path)
  loop_obj = os.listdir(class_dir_path)
  if pbar:
    loop_obj = tqdm(loop_obj)
  for file_name in loop_obj:
    if file_name.lower().endswith('.jpeg'):
      full_path = os.path.join(class_dir_path, file_name)
      with Image.open(full_path) as img:
        width, height = img.size
      records.append({'class_name': class_name, 'file_name': file_name, 'width': width, 'height': height})
  return records

def build_train_data_index(train_data_path, train_data_index_file, max_workers):
  class_dirs = [
    os.path.join(train_data_path, d)
    for d in os.listdir(train_data_path)
    if os.path.isdir(os.path.join(train_data_path, d))
  ]

  all_records = []
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_class_dir, class_dir) for class_dir in class_dirs]
    for future in as_completed(futures):
      result = future.result()
      all_records.extend(result)

  # for class_dir in class_dirs[0:2]:
  #   result = process_class_dir(class_dir, pbar=True)
  #   all_records.extend(result)

  df = pd.DataFrame(all_records)
  df.to_csv(train_data_index_file, index=False)  # index acts as image ID

def copy_and_rename_images(df, src_dir, dst_dir):
  os.makedirs(dst_dir, exist_ok=True)
  for idx in tqdm(range(len(df))):
    row = df.iloc[idx]
    src = os.path.join(src_dir, row['class_name'], row['file_name'])
    dst = os.path.join(dst_dir, row['image_name'])
    shutil.copy2(src, dst)

def build_dataset(ilscrv_path, output_dataset_path, resolution_threshold, train_size, valid_to_train_ratio):
  train_data_index_file = os.path.join(ilscrv_path, "train_data_index.csv")
  df = pd.read_csv(train_data_index_file)

  filtered = df[(df['width'] >= resolution_threshold) & (df['height'] >= resolution_threshold)]

  valid_size = int(train_size * valid_to_train_ratio)
  total_size = train_size + valid_size

  assert len(filtered) > total_size

  # Shuffle and split
  selected = filtered.sample(total_size, random_state=42).reset_index(drop=True)
  train_df = selected.iloc[:train_size].reset_index(drop=True)
  valid_df = selected.iloc[train_size:].reset_index(drop=True)

  num_train_digits = len(str(train_size-1))
  num_valid_digits = len(str(valid_size-1))

  train_df['image_name'] = [f"image_{i:0{num_train_digits}d}.jpg" for i in range(len(train_df))]
  valid_df['image_name'] = [f"image_{i:0{num_valid_digits}d}.jpg" for i in range(len(valid_df))]

  if os.path.exists(output_dataset_path):
    shutil.rmtree(output_dataset_path)
  os.makedirs(output_dataset_path, exist_ok = True)
  train_df.to_csv(os.path.join(output_dataset_path, "train_info.csv"), index=False)
  valid_df.to_csv(os.path.join(output_dataset_path, "valid_info.csv"), index=False)

  # Output dirs
  train_dir = os.path.join(output_dataset_path, 'train')
  valid_dir = os.path.join(output_dataset_path, 'valid')

  ilscrv_train_data_path = os.path.join(ilscrv_path, "ILSVRC/Data/CLS-LOC/train")
  copy_and_rename_images(train_df, ilscrv_train_data_path, train_dir)
  copy_and_rename_images(valid_df, ilscrv_train_data_path, valid_dir)

if __name__ == "__main__":
  dataset_base_path = "/capstor/scratch/cscs/ljiayong/datasets/ILSVRC-kaggle"
  train_data_path = os.path.join(dataset_base_path, "ILSVRC/Data/CLS-LOC/train")
  # '''
  # Step 1: Build index for all training images
  # '''
  # train_data_index_file = os.path.join(dataset_base_path, "train_data_index.csv")
  # max_workers = 64
  # build_train_data_index(train_data_path, train_data_index_file, max_workers)

  # '''
  # Step 2: Draw img size distribution
  # '''
  # train_data_index_file = os.path.join(dataset_base_path, "train_data_index.csv")
  # df = pd.read_csv(train_data_index_file)

  # # Extract width and height
  # widths = df['width']
  # heights = df['height']

  # # Plot 2D histogram (heatmap) of width vs height
  # plt.figure(figsize=(10, 8))
  # plt.hist2d(widths, heights, bins=100, cmap='plasma', norm='log')
  # plt.colorbar(label='Image Count')

  # # Draw lines at 256 and 512
  # plt.axvline(512, color='blue', linestyle='--', linewidth=1, label="512x512")
  # plt.axhline(512, color='blue', linestyle='--', linewidth=1)
  # plt.axvline(256, color='red', linestyle='--', linewidth=1, label="256x256")
  # plt.axhline(256, color='red', linestyle='--', linewidth=1)

  # # Labels and title
  # plt.xlabel('Width')
  # plt.ylabel('Height')
  # plt.title('Image Size Distribution in ImageNet Dataset')
  # plt.legend()
  # plt.grid(True)

  # output_path = f"./results/imagenet_image_size_dist.png"
  # plt.savefig(output_path, dpi=500, bbox_inches="tight")

  '''
  Step 3: Build dataset
  '''
  ilscrv_path = dataset_base_path
  output_dataset_base_path = "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets"
  resolution_threshold_lst = [256, 512]
  train_size = 50_000
  valid_to_train_ratio = 0.25

  # For 256: 1152197
  # For 512: 68410

  for resolution_threshold in resolution_threshold_lst:
    output_dataset_path = os.path.join(output_dataset_base_path, f"imagenet_{resolution_threshold}_{train_size}")
    print(f"[INFO] Starting to build dataset for resolution threshold {resolution_threshold} under {output_dataset_path}")
    build_dataset(ilscrv_path, output_dataset_path, resolution_threshold, train_size, valid_to_train_ratio)
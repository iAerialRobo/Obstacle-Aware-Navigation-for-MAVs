import cv2
import numpy as np
import os
import csv
import random
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Constants
TEXTONS_N_TEXTONS = 20
TEXTONS_PATCH_SIZE = 10
TEXTONS_ALPHA = 0.05
TEXTONS_DICTIONARY_PATH_DEFAULT = "/home/work/new/hunger"
DISTRIBUTION_OUTPUT_FOLDER_DEFAULT = "/home/work/paper"
IMAGE_PATH_DEFAULT = "/home/work/paper/NCIC9296.JPG"
WINDOW_SIZE = 40
PATCH_SIZE = TEXTONS_PATCH_SIZE
SAMPLES_PER_WINDOW = 20
WINDOW_STEP_SIZE = 8
TARGET_DISTRIBUTION_FILE = "/home/work/paper/obstcale.csv"

# Texton structure (Y, U, V components)
class Texton:
    def __init__(self):
        self.Y = np.zeros((TEXTONS_PATCH_SIZE, TEXTONS_PATCH_SIZE), dtype=np.float32)
        self.U = np.zeros((TEXTONS_PATCH_SIZE, TEXTONS_PATCH_SIZE // 2), dtype=np.float32)
        self.V = np.zeros((TEXTONS_PATCH_SIZE, TEXTONS_PATCH_SIZE // 2), dtype=np.float32)

# Global variables
dictionary = []
n_textons = TEXTONS_N_TEXTONS
patch_size = TEXTONS_PATCH_SIZE

# Helper function to ensure directory exists
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Load texton dictionary from binary file
def load_texton_dictionary(dictionary_path):
    filename = os.path.join(dictionary_path, "2010.bin")
    try:
        with open(filename, 'rb') as f:
            # Read dictionary dimensions
            loaded_n_textons = np.fromfile(f, dtype=np.int32, count=1)[0]
            loaded_patch_size = np.fromfile(f, dtype=np.int32, count=1)[0]

            if loaded_n_textons != n_textons or loaded_patch_size != patch_size:
                print("Loaded dictionary dimensions do not match current settings.")
                return False

            # Load dictionary data
            global dictionary
            dictionary = [Texton() for _ in range(n_textons)]

            for w in range(n_textons):
                dictionary[w].Y = np.fromfile(f, dtype=np.float32, count=patch_size * patch_size).reshape((patch_size, patch_size))
                dictionary[w].U = np.fromfile(f, dtype=np.float32, count=patch_size * (patch_size // 2)).reshape((patch_size, patch_size // 2))
                dictionary[w].V = np.fromfile(f, dtype=np.float32, count=patch_size * (patch_size // 2)).reshape((patch_size, patch_size // 2))

            print(f"Dictionary loaded from {filename}.")
            return True
    except Exception as e:
        print(f"Error loading dictionary file: {e}")
        return False

# Load distribution probabilities from CSV (适应之前保存的 CSV 格式：第一行标题 "Texton Index,Frequency")
def load_distribution_probabilities(distribution_file):
    P = [0.0] * n_textons
    try:
        with open(distribution_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行
            if len(header) < 2 or "Frequency" not in header[1]:
                print("CSV 文件标题格式不符合预期。")
                return []
            for row in reader:
                if row:
                    idx = int(row[0])
                    freq = float(row[1])
                    if 0 <= idx < n_textons:
                        P[idx] = freq
        print(f"Loaded distribution probabilities from {distribution_file}.")
        if len(P) != n_textons:
            print(f"Error: Expected {n_textons} probabilities, but found {len(P)}.")
            return []
        return P
    except Exception as e:
        print(f"Error loading distribution file: {e}")
        return []

# Function to process a single window，使用向量化方式处理窗口内所有 patch
def process_window(args):
    (y_pos, x_pos, Y, U_subsampled, V_subsampled, P, dictionary, rng_seed) = args
    rng = random.Random(rng_seed + y_pos * 10000 + x_pos)
    
    # 定义窗口区域
    window_Y = Y[y_pos:y_pos + WINDOW_SIZE, x_pos:x_pos + WINDOW_SIZE]
    window_U = U_subsampled[y_pos:y_pos + WINDOW_SIZE, x_pos // 2:x_pos // 2 + WINDOW_SIZE // 2]
    window_V = V_subsampled[y_pos:y_pos + WINDOW_SIZE, x_pos // 2:x_pos // 2 + WINDOW_SIZE // 2]
    
    # 预先构造字典矩阵，便于后续向量化计算
    dict_Y = np.stack([t.Y for t in dictionary], axis=0)  # shape: (n_textons, PATCH_SIZE, PATCH_SIZE)
    dict_U = np.stack([t.U for t in dictionary], axis=0)  # shape: (n_textons, PATCH_SIZE, PATCH_SIZE//2)
    dict_V = np.stack([t.V for t in dictionary], axis=0)  # shape: (n_textons, PATCH_SIZE, PATCH_SIZE//2)
    
    # 随机生成 SAMPLES_PER_WINDOW 个 patch 的左上角坐标
    max_offset = WINDOW_SIZE - PATCH_SIZE
    patch_coords = [(rng.randint(0, max_offset), rng.randint(0, max_offset)) for _ in range(SAMPLES_PER_WINDOW)]
    
    # 提取所有 patch（列表转为 numpy 数组）
    patches_Y = np.array([window_Y[y:y+PATCH_SIZE, x:x+PATCH_SIZE] for (y, x) in patch_coords], dtype=np.float32)
    # 对于 U 和 V，水平方向坐标需要除以2
    patches_U = np.array([window_U[y:y+PATCH_SIZE, (x//2):(x//2)+PATCH_SIZE//2] for (y, x) in patch_coords], dtype=np.float32)
    patches_V = np.array([window_V[y:y+PATCH_SIZE, (x//2):(x//2)+PATCH_SIZE//2] for (y, x) in patch_coords], dtype=np.float32)
    
    # 计算每个 patch 与每个 texton 的距离（向量化）
    # patches_Y: (SAMPLES_PER_WINDOW, PATCH_SIZE, PATCH_SIZE)
    # dict_Y: (n_textons, PATCH_SIZE, PATCH_SIZE)
    # 计算差值后平方求和：结果 shape (SAMPLES_PER_WINDOW, n_textons)
    diff_Y = patches_Y[:, None, :, :] - dict_Y[None, :, :, :]
    distances_Y = np.sum(diff_Y ** 2, axis=(2, 3))
    
    diff_U = patches_U[:, None, :, :] - dict_U[None, :, :, :]
    distances_U = np.sum(diff_U ** 2, axis=(2, 3))
    
    diff_V = patches_V[:, None, :, :] - dict_V[None, :, :, :]
    distances_V = np.sum(diff_V ** 2, axis=(2, 3))
    
    total_distances = distances_Y + distances_U + distances_V  # shape: (SAMPLES_PER_WINDOW, n_textons)
    
    # 对每个 patch，找到最小距离对应的 texton 索引
    best_textons = np.argmin(total_distances, axis=1)
    
    # 统计各个 texton 的出现次数
    texton_counts = np.bincount(best_textons, minlength=n_textons)
    
    # 计算 T 值
    T = np.sum(texton_counts * np.array(P))
    
    return (y_pos, x_pos, texton_counts, T)

# Extract distributions and save results
def distribution_extraction(image, Y, U_subsampled, V_subsampled, image_name, output_folder, P):
    window_size = WINDOW_SIZE
    step_size = WINDOW_STEP_SIZE
    max_x = Y.shape[1] - window_size
    max_y = Y.shape[0] - window_size

    csv_filename = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_Distributions.csv")
    
    tasks = []
    rng_seed = 42  # 固定种子保证可重复性
    for y_pos in range(0, max_y, step_size):
        for x_pos in range(0, max_x, step_size):
            tasks.append((y_pos, x_pos, Y, U_subsampled, V_subsampled, P, dictionary, rng_seed))

    results = []
    window_T_pairs = []
    output_image = image.copy()

    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_window = {executor.submit(process_window, task): (task[0], task[1]) for task in tasks}
        for future in tqdm(as_completed(future_to_window), total=len(future_to_window), desc="Processing Windows"):
            y_pos, x_pos = future_to_window[future]
            try:
                y, x, texton_counts, T = future.result()
                window_T_pairs.append(((x, y), T))
                results.append((x, y, texton_counts, T))
            except Exception as e:
                print(f"Error processing window at ({x_pos}, {y_pos}): {e}")

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Window_ID", "X", "Y"] + [f"Texton_{i}" for i in range(n_textons)] + ["T_Value"])
        max_T = max([pair[1] for pair in window_T_pairs], default=1.0)

        for idx, (x, y, texton_counts, T) in enumerate(results, start=1):
            csv_writer.writerow([f"Window_{idx}", x, y] + list(texton_counts) + [T])
            normalized_T = T / max(1.0, max_T)
            blue = int(255 * (1 - normalized_T))
            green = 255
            red = int(255 * (1 - normalized_T))
            color = (blue, green, red)
            cv2.rectangle(output_image, (x, y), (x + window_size, y + window_size), color, -1)

    print(f"Saved distributions to {csv_filename}")
    output_image_filename = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_Marked.png")
    cv2.imwrite(output_image_filename, output_image)
    print(f"Saved marked image to {output_image_filename}")

# Main function
def main():
    dictionary_path = TEXTONS_DICTIONARY_PATH_DEFAULT
    output_folder = DISTRIBUTION_OUTPUT_FOLDER_DEFAULT
    image_path = IMAGE_PATH_DEFAULT
    distribution_file = TARGET_DISTRIBUTION_FILE

    if not load_texton_dictionary(dictionary_path):
        print("Failed to load texton dictionary. Exiting.")
        return

    ensure_directory_exists(output_folder)

    P = load_distribution_probabilities(distribution_file)
    if len(P) != n_textons:
        print("Mismatch in number of Textons and distribution probabilities. Exiting.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}. Exiting.")
        return

    image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(image_YUV)
    U_subsampled = cv2.resize(U, (U.shape[1] // 2, U.shape[0]), interpolation=cv2.INTER_LINEAR)
    V_subsampled = cv2.resize(V, (V.shape[1] // 2, V.shape[0]), interpolation=cv2.INTER_LINEAR)

    distribution_extraction(image, Y, U_subsampled, V_subsampled, os.path.basename(image_path), output_folder, P)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import random
import time
import glob
import struct
from collections import namedtuple

# 定义常量
TEXTONS_N_TEXTONS = 30
TEXTONS_PATCH_SIZE =10
TEXTONS_ALPHA = 0.05
TEXTONS_DICTIONARY_PATH = "/home/work/new/hunger"
FRAME_DIR_PATH = "/home/work/train hunger"
TEXTONS_N_SAMPLES = 1000 # 每帧提取的样本数
DICTIONARY_FILENAME = "3010.bin"  # 使用.bin保存词典

# 定义Texton结构
Texton = namedtuple('Texton', ['Y', 'U', 'V'])

# 全局变量
dictionary = []
learned_samples = 0
dictionary_initialized = False
n_textons = TEXTONS_N_TEXTONS
patch_size = TEXTONS_PATCH_SIZE
alpha = TEXTONS_ALPHA

def ensure_directory_exists(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_dictionary():
    """可视化词典中的所有texton"""
    if len(dictionary) < n_textons:
        print(f"词典大小 ({len(dictionary)}) 小于 n_textons ({n_textons})。")
        return

    num_images = n_textons
    num_cols = 10
    num_rows = (num_images + num_cols - 1) // num_cols
    patch_display_size = 100
    gap = 10

    display_image = np.zeros(
        ((patch_display_size + gap) * num_rows - gap, (patch_display_size + gap) * num_cols - gap, 3),
        dtype=np.uint8
    )

    for idx in range(num_images):
        patch_Y = dictionary[idx].Y
        patch_U = dictionary[idx].U
        patch_V = dictionary[idx].V

        # 上采样U和V到与Y相同的分辨率
        U_upsampled = cv2.resize(patch_U, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        V_upsampled = cv2.resize(patch_V, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

        # 合并Y, U, V为一个YUV图像
        yuv_image = cv2.merge([patch_Y, U_upsampled, V_upsampled])

        # 转换为8位图像，然后转换为BGR
        yuv_image_8u = cv2.convertScaleAbs(yuv_image)
        bgr_image = cv2.cvtColor(yuv_image_8u, cv2.COLOR_YUV2BGR)

        # 调整大小以便显示
        resized_img = cv2.resize(bgr_image, (patch_display_size, patch_display_size))

        # 计算在显示网格中的位置
        row_pos = idx // num_cols
        col_pos = idx % num_cols
        startX = col_pos * (patch_display_size + gap)
        startY = row_pos * (patch_display_size + gap)

        # 复制到 display_image
        display_image[startY:startY + patch_display_size, startX:startX + patch_display_size] = resized_img

    # 显示词典
    cv2.imshow("Texton Dictionary - Color Visualization", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存可视化图像
    save_path = os.path.join(TEXTONS_DICTIONARY_PATH, "visualization")
    ensure_directory_exists(save_path)
    filename = os.path.join(save_path, "Texton_Dictionary_Color.png")
    cv2.imwrite(filename, display_image)
    print(f"Texton词典可视化已保存为 '{filename}'。")

def textons_init(image_files):
    """初始化词典，从第一张图像中随机选择n_textons个patch作为初始词典"""
    global dictionary, patch_size, learned_samples, dictionary_initialized

    if patch_size % 2 == 1:
        patch_size += 1
        print(f"patch_size 是奇数，已增加到 {patch_size}")

    if not image_files:
        raise ValueError("未提供用于初始化词典的图像文件。")

    # 使用第一张图像进行初始化
    init_image = image_files[0]
    frame = cv2.imread(init_image)
    if frame is None:
        raise ValueError(f"无法加载图像: {init_image}")

    frame_YUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(frame_YUV)

    # 对U和V进行水平下采样（YUV422）
    U_subsampled = cv2.resize(U, (U.shape[1] // 2, U.shape[0]), interpolation=cv2.INTER_LINEAR)
    V_subsampled = cv2.resize(V, (V.shape[1] // 2, V.shape[0]), interpolation=cv2.INTER_LINEAR)

    height, width = Y.shape

    dictionary = []
    for _ in range(n_textons):
        # 随机选择patch的左上角坐标
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)

        # 提取Y, U, V通道的patch
        patch_Y = Y[y:y + patch_size, x:x + patch_size].astype(np.float32)
        patch_U = U_subsampled[y:y + patch_size, x // 2:x // 2 + patch_size // 2].astype(np.float32)
        patch_V = V_subsampled[y:y + patch_size, x // 2:x // 2 + patch_size // 2].astype(np.float32)

        dictionary.append(Texton(patch_Y, patch_U, patch_V))

    learned_samples = n_textons
    dictionary_initialized = True
    print("词典已使用第一张图像中的patch初始化。")

def load_texton_dictionary():
    """加载已保存的词典"""
    global dictionary, learned_samples, dictionary_initialized
    filename = os.path.join(TEXTONS_DICTIONARY_PATH, DICTIONARY_FILENAME)
    if not os.path.exists(filename):
        print(f"词典文件 '{filename}' 不存在。")
        return False
    with open(filename, 'rb') as f:
        # 读取n_textons和patch_size
        data = f.read(8)  # 4字节n_textons, 4字节patch_size
        n_textons_loaded, patch_size_loaded = struct.unpack('ii', data)

        if n_textons_loaded != n_textons or patch_size_loaded != patch_size:
            print("词典参数与当前设置不匹配。")
            return False

        dictionary = []
        for _ in range(n_textons_loaded):
            # 读取Y分量
            Y = np.zeros((patch_size, patch_size), dtype=np.float32)
            for i in range(patch_size):
                for j in range(patch_size):
                    Y[i, j] = struct.unpack('f', f.read(4))[0]

            # 读取U分量
            U = np.zeros((patch_size, patch_size // 2), dtype=np.float32)
            for i in range(patch_size):
                for j in range(patch_size // 2):
                    U[i, j] = struct.unpack('f', f.read(4))[0]

            # 读取V分量
            V = np.zeros((patch_size, patch_size // 2), dtype=np.float32)
            for i in range(patch_size):
                for j in range(patch_size // 2):
                    V[i, j] = struct.unpack('f', f.read(4))[0]

            dictionary.append(Texton(Y, U, V))

    print(f"词典已从 '{filename}' 加载。")
    learned_samples = n_textons
    dictionary_initialized = True
    return True

def save_texton_dictionary():
    """保存当前词典到文件"""
    filename = os.path.join(TEXTONS_DICTIONARY_PATH, DICTIONARY_FILENAME)
    ensure_directory_exists(TEXTONS_DICTIONARY_PATH)
    with open(filename, 'wb') as f:
        # 写入n_textons和patch_size
        f.write(struct.pack('ii', n_textons, patch_size))
        # 写入每个texton的Y, U, V
        for texton in dictionary:
            # 写入Y分量
            for i in range(patch_size):
                for j in range(patch_size):
                    f.write(struct.pack('f', texton.Y[i, j]))
            # 写入U分量
            for i in range(patch_size):
                for j in range(patch_size // 2):
                    f.write(struct.pack('f', texton.U[i, j]))
            # 写入V分量
            for i in range(patch_size):
                for j in range(patch_size // 2):
                    f.write(struct.pack('f', texton.V[i, j]))
    print(f"词典已保存到 '{filename}'。")

def DictionaryTrainingYUV(Y, U_subsampled, V_subsampled):
    """使用YUV图像数据训练词典"""
    global dictionary, learned_samples, dictionary_initialized
    height, width = Y.shape

    for _ in range(TEXTONS_N_SAMPLES):
        # 随机选择patch的左上角坐标
        if width <= patch_size or height <= patch_size:
            print("帧大小小于patch大小，跳过样本提取。")
            break
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)

        # 提取Y, U, V通道的patch
        patch_Y = Y[y:y + patch_size, x:x + patch_size].astype(np.float32)
        patch_U = U_subsampled[y:y + patch_size, x // 2:x // 2 + patch_size // 2].astype(np.float32)
        patch_V = V_subsampled[y:y + patch_size, x // 2:x // 2 + patch_size // 2].astype(np.float32)

        # 计算每个texton的距离
        # 使用numpy高效计算
        texton_Y = np.array([texton.Y for texton in dictionary])
        texton_U = np.array([texton.U for texton in dictionary])
        texton_V = np.array([texton.V for texton in dictionary])

        distances = np.sum((texton_Y - patch_Y) ** 2, axis=(1,2)) + \
                    np.sum((texton_U - patch_U) ** 2, axis=(1,2)) + \
                    np.sum((texton_V - patch_V) ** 2, axis=(1,2))

        # 找到距离最小的texton
        assignment = np.argmin(distances)

        # 更新被分配的texton
        dictionary[assignment] = Texton(
            dictionary[assignment].Y + alpha * (patch_Y - dictionary[assignment].Y),
            dictionary[assignment].U + alpha * (patch_U - dictionary[assignment].U),
            dictionary[assignment].V + alpha * (patch_V - dictionary[assignment].V)
        )

        learned_samples += 1
        if learned_samples % 1000 == 0:
            print(f"已处理 {learned_samples} 个样本。")

def textons_stop():
    """停止文本子处理并释放资源的函数"""
    global dictionary, learned_samples, dictionary_initialized
    # 清除词典
    dictionary.clear()
    learned_samples = 0
    dictionary_initialized = False
    print("资源已释放。")

def main():
    """主函数，控制词典的加载、初始化、训练和保存"""
    global patch_size, learned_samples, dictionary_initialized

    # 设置随机种子
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    if patch_size % 2 == 1:
        patch_size += 1
        print(f"patch_size 是奇数，已增加到 {patch_size}")

    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(FRAME_DIR_PATH, "*.JPG")) +
                        glob.glob(os.path.join(FRAME_DIR_PATH, "*.jpeg")) +
                        glob.glob(os.path.join(FRAME_DIR_PATH, "*.png")) +
                        glob.glob(os.path.join(FRAME_DIR_PATH, "*.bmp")) +
                        glob.glob(os.path.join(FRAME_DIR_PATH, "*.tiff")) +
                        glob.glob(os.path.join(FRAME_DIR_PATH, "*.tif")))

    if not image_files:
        print(f"在目录 {FRAME_DIR_PATH} 中未找到任何图像文件。")
        return

    # 检查是否存在已保存的词典文件
    filename = os.path.join(TEXTONS_DICTIONARY_PATH, DICTIONARY_FILENAME)
    if os.path.exists(filename):
        print("找到已保存的词典文件。正在加载...")
        loaded = load_texton_dictionary()
        if not loaded:
            print("加载词典失败。正在初始化新的词典...")
            textons_init(image_files)
    else:
        print("未找到已保存的词典文件。正在初始化词典...")
        textons_init(image_files)

    print(f"开始处理 {len(image_files)} 张图像...")

    # 遍历每一张图像进行训练
    for idx, image_path in enumerate(image_files):
        print(f"正在处理图像 {idx + 1}/{len(image_files)}: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法加载图像: {image_path}")
            continue

        # 将图像转换为YUV格式
        frame_YUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # 分离Y, U, V通道
        Y, U, V = cv2.split(frame_YUV)

        # 对U和V进行水平下采样（YUV422）
        U_subsampled = cv2.resize(U, (U.shape[1] // 2, U.shape[0]), interpolation=cv2.INTER_LINEAR)
        V_subsampled = cv2.resize(V, (V.shape[1] // 2, V.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 训练词典
        DictionaryTrainingYUV(Y, U_subsampled, V_subsampled)

    # 训练完成后保存词典并可视化
    save_texton_dictionary()
    visualize_dictionary()
    print("训练完成并已保存词典。")
    textons_stop()

if __name__ == "__main__":
    main()

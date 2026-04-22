import os
import sys
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from config_module import config_file

config = config_file.config_class()

class EVEDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []

        # 获取所有参与者的目录
        participants = [os.path.join(root_dir, f"{split}{i:02d}") for i in range(1, 40) if split == "train" or (split == "val" and i <= 5) or (split == "test" and i <= 6)]
        
        for participant in participants:
            all_items = os.listdir(participant)
            stimulus_name_list = [item for item in all_items if os.path.isdir(os.path.join(participant, item))]
            stimulus_path_list = [os.path.join(participant, item) for item in stimulus_name_list][:config.EVE_dataset.num_dirs_per_person]
            for stimulus_path in stimulus_path_list:
                for camera in ["basler", "webcam_c", "webcam_l", "webcam_r"]:
                    video_file = os.path.join(stimulus_path, f"{camera}.mp4")
                    h5_file = os.path.join(stimulus_path, f"{camera}.h5")
                    if os.path.exists(video_file) and os.path.exists(h5_file):
                        with h5py.File(h5_file, "r") as f:
                            num_frames = f["face_g_tobii"]["data"].shape[0]
                            for frame_idx in range(0, num_frames, num_frames // config.EVE_dataset.num_frames_per_video):
                                if (f["face_g_tobii"]["validity"][frame_idx]) and (f["face_R"]["validity"][frame_idx]):
                                    self.data.append((video_file, h5_file, frame_idx))
            #print(f"{split} set: Having found {count} samples in {participant}")
        #print(f"\n\nTotal {len(self.data)} samples in {split} set\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, h5_file, frame_idx = self.data[idx]

        # 读取HDF5文件中的数据
        with h5py.File(h5_file, "r") as f:
            g_tobii = f["face_PoG_tobii"]["data"][frame_idx]  # 使用面部的归一化视线方向

        return g_tobii

# 绘制预测点和注视点
def plot_points(points, screen_width, screen_height):
    # 创建图形
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 设置背景为白色
    ax.set_facecolor('white')

    # 绘制屏幕边界
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
    plt.axvline(x=screen_width, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=screen_height, color='black', linestyle='-', linewidth=2)

    # 绘制预测点（蓝色点）
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=20, label='Train Points')

    # 设置坐标轴范围
    plt.xlim(-screen_width * 0.1, screen_width * 1.1)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0.1, screen_height * 1.1)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.25))  # 将图例放置在图像右侧

    # 设置标题和坐标轴标签
    plt.title('Predicted Points with Gaze Point')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 显示图像
    plt.show()

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# 创建数据集并处理数据
dataset = EVEDataset(root_dir="F:/EVE_dataset/eve_dataset", split="train")

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

train_PoG = np.array([])

# 处理数据并显示进度条
for data in tqdm(dataloader, desc="Processing data", total=len(dataset)):
    g_tobii = data.numpy()
    if len(train_PoG) == 0:
        train_PoG = g_tobii
    else:
        train_PoG = np.vstack((train_PoG, g_tobii))

plot_points(train_PoG, SCREEN_WIDTH, SCREEN_HEIGHT)
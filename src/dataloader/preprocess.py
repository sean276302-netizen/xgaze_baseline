import os
import sys
import h5py
import cv2
import numpy as np
import torch
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(script_dir, '..', '..'))

from config_module import config_file
from ultralytics import YOLO
from face_detection import detect_face

yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')
config = config_file.config_class()

num_dirs_per_participant = 30  # 每个参与者的视频数量
num_frames_per_video = 25  # 每个视频的帧数

def crop_head(image, output_size=(224, 224), padding_ratio=config.train.face_extending_ratio):
    if image is None:
        raise ValueError("无法加载图像，请检查路径是否正确")

    detections = detect_face.detect_face_yolo_for_face_model(image, yolo)

    # 如果没有检测到人脸，返回 None
    if len(detections) == 0:
        return None

    # 选择置信度最高的检测结果
    best_detection = detections[0]

    # 解析检测结果
    x1, y1, x2, y2 = best_detection
    # 获得宽高
    width = x2 - x1
    height = y2 - y1

    # 增加边界范围
    delta_w = int(width * padding_ratio)
    delta_h = int(height * padding_ratio)

    # 调整边界
    x1_new = int(max(0, x1 - delta_w))
    y1_new = int(max(0, y1 - delta_h))
    x2_new = int(min(image.shape[1], x2 + delta_w))
    y2_new = int(min(image.shape[0], y2 + delta_h))

    # 裁切图像
    head_image = image[y1_new:y2_new, x1_new:x2_new]

    # 调整裁切图像的大小
    head_image = cv2.resize(head_image, output_size)

    return head_image

class EVEDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.participant_h5_files = {}  # 用于保存每个参与者的 h5 文件对象
        self.lock = threading.Lock()  # 线程锁，确保文件操作安全
        dataset_dir = "/mnt/e/EVE_dataset_processed"

        # 获取所有参与者的目录
        participants = [os.path.join(root_dir, f"{split}{i:02d}") for i in range(1, 36) 
                       if split == "train" or (split == "val" and i <= 5) or (split == "test" and i <= 4)][:1]
        
        participants = tqdm(participants, desc=f"Loading {split} participants")
        for participant_dir in participants:
            participant_idx = 0
            participant_id = os.path.basename(participant_dir)  # 获取参与者编号（例如 train01）
            h5_dir = os.path.join(dataset_dir, split)  # 获取 h5 文件的路径
            if not os.path.exists(h5_dir):
                os.makedirs(h5_dir)
            h5_path = os.path.join(h5_dir, f"{participant_id}.h5")
            self.participant_h5_files[participant_id] = h5py.File(h5_path, 'w')  # 创建 h5 文件

            # 遍历当前参与者的数据
            all_items = os.listdir(participant_dir)
            stimulus_name_list = [item for item in all_items if os.path.isdir(os.path.join(participant_dir, item))]
            stimulus_path_list = [os.path.join(participant_dir, item) for item in stimulus_name_list][:num_dirs_per_participant][:1]
            for stimulus_path in stimulus_path_list:
                for camera in ["basler", "webcam_c", "webcam_l", "webcam_r"][:1]:
                    video_file = os.path.join(stimulus_path, f"{camera}.mp4")
                    h5_file = os.path.join(stimulus_path, f"{camera}.h5")
                    if os.path.exists(video_file) and os.path.exists(h5_file):
                        with h5py.File(h5_file, "r") as f:
                            num_frames = f["face_g_tobii"]["data"].shape[0]
                            count = 0
                            for frame_idx in range(num_frames)[:1]:
                                if (f["face_g_tobii"]["validity"][frame_idx]) and (f["face_R"]["validity"][frame_idx]):
                                    participant_idx += 1
                                    self.data.append((video_file, h5_file, frame_idx, participant_id, participant_idx - 1))  # 增加participant_id
                                    count += 1
                                    if count >= num_frames_per_video: break  # 限制每个视频的帧数为 25

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, h5_file, frame_idx, participant_id, participant_idx = self.data[idx]

        # 读取视频帧
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_file}")

        frame = crop_head(frame)
        if frame is None:
            raise ValueError("No face detected in the frame")
        frame = np.array(frame, dtype=np.float32)

        # 读取 HDF5 文件中的数据
        with h5py.File(h5_file, "r") as f:
            g_tobii = f["face_g_tobii"]["data"][frame_idx]
            R = f["face_R"]["data"][frame_idx]

        pitch, yaw = g_tobii
        gaze_vector = self.spherical_to_vector(pitch, yaw)
        R = torch.tensor(R, dtype=torch.float32)
        gaze_vector = torch.tensor(gaze_vector, dtype=torch.float32)
        
        # 使用 PyTorch 计算逆矩阵和矩阵乘法
        original_gaze_vector = torch.linalg.inv(R) @ gaze_vector
        original_pitch, original_yaw = self.vector_to_spherical(original_gaze_vector.detach().cpu().numpy())

        initial_size = num_frames_per_video * 4 * num_dirs_per_participant  # 初始数据集大小

        # 将数据保存到对应的 h5 文件中
        h5_file = self.participant_h5_files.get(participant_id, None)
        if h5_file:
            with self.lock:  # 使用线程锁确保文件操作安全
                if 'images' not in h5_file:
                    h5_file.create_dataset(
                        'images', 
                        shape=(initial_size, *frame.shape),
                        maxshape=(None, *frame.shape),
                        chunks=True,
                        dtype=np.float32
                    )
                    h5_file.create_dataset(
                        'labels', 
                        shape=(initial_size, 2),
                        maxshape=(None, 2),
                        chunks=True,
                        dtype=np.float32
                    )
                
                # 扩展 dataset 的大小
                image_dataset = h5_file['images']
                label_dataset = h5_file['labels']
                
                # 保存数据
                image_dataset[participant_idx] = np.array(frame)
                label_dataset[participant_idx] = np.array([original_pitch, original_yaw])

        return frame, torch.tensor([original_pitch, original_yaw], dtype=torch.float32)

    @staticmethod
    def spherical_to_vector(pitch, yaw):
        x = np.cos(pitch) * np.cos(yaw)
        y = np.cos(pitch) * np.sin(yaw)
        z = np.sin(pitch)
        return [x, y, z]

    @staticmethod
    def vector_to_spherical(vector):
        x, y, z = vector
        pitch = np.arcsin(z)
        yaw = np.arctan2(y, x)
        return pitch, yaw
    
    '''@staticmethod
    def spherical_to_vector(pitch, yaw):
        x = cp.cos(pitch) * cp.cos(yaw)
        y = cp.cos(pitch) * cp.sin(yaw)
        z = cp.sin(pitch)
        return cp.asarray([x, y, z], dtype=cp.float32)

    @staticmethod
    def vector_to_spherical(vector):
        # 将输入向量转换为 PyTorch 张量
        vector = torch.tensor(vector, dtype=torch.float32, device='cuda')
        
        # 在 GPU 上进行计算
        x, y, z = vector
        pitch = torch.arcsin(z)
        yaw = torch.arctan2(y, x)
        
        # 返回标量值
        return pitch.item(), yaw.item()'''

    def close_h5_files(self):
        """关闭所有 h5 文件"""
        for h5_file in self.participant_h5_files.values():
            h5_file.close()

for split in ["train", "val", "test"][:1]:
    # 创建数据集并处理数据
    dataset = EVEDataset(root_dir="/mnt/e/EVE_dataset/eve_dataset", split=split)

    # 使用多线程加速数据处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # 确保在程序结束时关闭所有 h5 文件
        try:
            # 处理数据并显示进度条
            for data in tqdm(dataloader, desc="Processing data", total=len(dataset)):
                pass  # 数据存储已经在 dataset 的 __getitem__ 方法中完成
        finally:
            dataset.close_h5_files()
import os
import sys
import cv2
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))

from config_module import config_file
from ultralytics import YOLO
from face_detection import detect_face

yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')
config = config_file.config_class()

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
    x1_new = max(0, x1 - delta_w)
    y1_new = max(0, y1 - delta_h)
    x2_new = min(image.shape[1], x2 + delta_w)
    y2_new = min(image.shape[0], y2 + delta_h)

    # 确保边界是整数类型
    x1_new = int(x1_new)
    y1_new = int(y1_new)
    x2_new = int(x2_new)
    y2_new = int(y2_new)

    # 裁切图像
    head_image = image[y1_new:y2_new, x1_new:x2_new]

    # 调整裁切图像的大小
    head_image = cv2.resize(head_image, output_size)

    # 返回裁切后的图像
    return head_image

class EVEDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []

        # 获取所有参与者的目录
        participants = [os.path.join(root_dir, f"{split}{i:02d}") for i in range(1, 36) if split == "train" or (split == "val" and i <= 5) or (split == "test" and i <= 4)]
        
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
                            for frame_idx in range(num_frames)[:config.EVE_dataset.num_frames_per_video]:
                                if (f["face_g_tobii"]["validity"][frame_idx]) and (f["face_R"]["validity"][frame_idx]):
                                    self.data.append((video_file, h5_file, frame_idx))
            #print(f"{split} set: Having found {count} samples in {participant}")
        #print(f"\n\nTotal {len(self.data)} samples in {split} set\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, h5_file, frame_idx = self.data[idx]

        # 读取视频帧
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = crop_head(frame)
        cap.release()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_file}")

        # 读取HDF5文件中的数据
        with h5py.File(h5_file, "r") as f:
            g_tobii = f["face_g_tobii"]["data"][frame_idx]  # 使用面部的归一化视线方向
            R = f["face_R"]["data"][frame_idx]  # 使用面部的旋转矩阵

        # 将 g_tobii (pitch, yaw) 转换为三维向量
        pitch, yaw = g_tobii
        gaze_vector = self.spherical_to_vector(pitch, yaw)

        # 应用旋转矩阵
        R = torch.tensor(R, dtype=torch.float32)
        gaze_vector = torch.tensor(gaze_vector, dtype=torch.float32)
        original_gaze_vector = torch.matmul(np.linalg.inv(R), gaze_vector)

        # 将三维向量转换回 pitch 和 yaw
        original_pitch, original_yaw = self.vector_to_spherical(original_gaze_vector)

        # 转换为CHW格式并应用预处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)  # 转换为PIL Image
        if self.transform:
            frame = self.transform(frame)

        return frame, torch.tensor([original_pitch, original_yaw], dtype=torch.float32)

    @staticmethod
    def spherical_to_vector(pitch, yaw):
        # 将 pitch 和 yaw 转换为三维向量
        x = np.cos(pitch) * np.cos(yaw)
        y = np.cos(pitch) * np.sin(yaw)
        z = np.sin(pitch)
        return [x, y, z]

    @staticmethod
    def vector_to_spherical(vector):
        # 将三维向量转换为 pitch 和 yaw
        x, y, z = vector
        pitch = np.arcsin(z)
        yaw = np.arctan2(y, x)
        return pitch, yaw

class XGazeDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []

        if split == "train": participants = [os.path.join(root_dir, "train")]
        elif split == "val": participants = [os.path.join(root_dir, "val")]
        elif split == "test": participants = [os.path.join(root_dir, "test")]
        else: raise ValueError("Invalid split")

        for participant in participants:
            for stimulus in os.listdir(participant):
                stimulus_path = os.path.join(participant, stimulus)
                with h5py.File(stimulus_path, "r") as f:
                    num_frames = f["face_patch"].shape[0]
                    for frame_idx in range(num_frames)[:config.XGaze_dataset.num_frames_per_person]:
                        self.data.append((stimulus_path, frame_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stimulus_path, frame_idx = self.data[idx]

        with h5py.File(stimulus_path, "r") as f:
            frame = f["face_patch"][frame_idx]
            pitch, yaw = f["face_gaze"][frame_idx]

        # 转换为CHW格式并应用预处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)  # 转换为PIL Image
        if self.transform:
            frame = self.transform(frame)

        return frame, torch.tensor([pitch, yaw], dtype=torch.float32)


# 在dataloader的train transforms中添加更多增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

'''train_transform = transforms.Compose([
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机旋转
    transforms.RandomRotation(degrees=15),
    # 调整亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 随机裁剪
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # 随机仿射变换
    transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)),
    # 转换为 Tensor
    transforms.ToTensor(),
    # 高斯模糊
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # 随机擦除
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    # 归一化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])'''

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if config.train.dataset == "EVE":
    # 数据加载器
    train_dataset = EVEDataset(root_dir=config.EVE_dataset.data_path, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)

    val_dataset = EVEDataset(root_dir=config.EVE_dataset.data_path, split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

    test_dataset = EVEDataset(root_dir=config.EVE_dataset.data_path, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False, num_workers=config.test.num_workers)

elif config.train.dataset == "XGaze":
    # 数据加载器
    train_dataset = XGazeDataset(root_dir=config.XGaze_dataset.data_path, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)

    val_dataset = XGazeDataset(root_dir=config.XGaze_dataset.data_path, split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

    test_dataset = XGazeDataset(root_dir=config.XGaze_dataset.data_path, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False, num_workers=config.test.num_workers)

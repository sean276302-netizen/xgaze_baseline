import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from modules import resnet

class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet.resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze

class gaze_network_v2(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network_v2, self).__init__()
        self.gaze_network = resnet.resnet50(pretrained=True)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 2048), nn.Sigmoid()
        )

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # 调整后的localization网络，保留更多空间信息
        '''self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),  # 输出10x1x1
            nn.AdaptiveAvgPool2d((3, 3)),      # 调整为10x3x3
            nn.ReLU(True)
        )'''
        # 简化的 localization 网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # 减少 kernel size 和 stride
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),  # 输出 tensor: 10x3x3
            nn.AdaptiveAvgPool2d((3, 3)),  # 确保输出为 3x3
            nn.ReLU(True)
        )
        # 全连接层输出8个参数，用于构造3x3透视变换矩阵
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 8)
        )
        # 初始化参数为恒等变换
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    def forward(self, x):
        # 提取空间变换参数
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        batch_size = theta.size(0)
        
        # 构造3x3透视变换矩阵
        theta_mat = torch.zeros(batch_size, 3, 3, device=x.device)
        theta_mat[:, 0, :] = theta[:, 0:3]    # 第一行参数
        theta_mat[:, 1, :] = theta[:, 3:6]    # 第二行参数
        theta_mat[:, 2, 0:2] = theta[:, 6:8]  # 第三行前两个参数
        theta_mat[:, 2, 2] = 1.0              # 第三行第三个元素固定为1

        # 生成归一化网格
        h, w = x.shape[2], x.shape[3]
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                       torch.linspace(-1, 1, w, device=x.device),
                                       indexing='ij')
        ones = torch.ones_like(x_grid)
        base_grid = torch.stack([x_grid, y_grid, ones], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 应用透视变换矩阵
        transformed = torch.matmul(base_grid.view(batch_size, -1, 3), theta_mat.transpose(1, 2))
        transformed = transformed.view(batch_size, h, w, 3)
        
        # 归一化坐标
        z = transformed[..., 2].clamp(min=1e-6)  # 防止除以零
        x_trans = transformed[..., 0] / z
        y_trans = transformed[..., 1] / z
        new_grid = torch.stack([x_trans, y_trans], dim=-1)

        # 应用网格采样
        x = F.grid_sample(x, new_grid, align_corners=False)
        return x, theta_mat  # 返回变换后的图像和变换矩阵

class gaze_network_STN(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network_STN, self).__init__()
        self.stn = STN()
        # 加载预训练ResNet50
        self.gaze_network = resnet.resnet50(pretrained=True)
        # 调整最后的全连接层
        self.gaze_fc = nn.Sequential(
            #nn.Dropout(0.5),  # 添加Dropout层，丢弃率为0.5
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        # STN变换并获取变换矩阵
        x, theta_mat = self.stn(x)
        # 特征提取
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        # 预测视线方向
        gaze = self.gaze_fc(feature)
        # 使用camera_matrix和theta_mat进行视线方向还原
        true_gaze = self._restore_gaze(gaze, theta_mat)
        return true_gaze

    def _restore_gaze(self, gaze, H):
        """
        使用变换矩阵还原视线方向
        参数：
            gaze: 预测的视线方向 (pitch, yaw) (batch, 2)
            H: 透视变换矩阵 (batch, 3, 3)
        返回：
            restored_gaze: 还原后的真实视线方向 (batch, 2)
        """
        # 将视线方向 (pitch, yaw) 转换为三维单位向量
        pitch, yaw = gaze[:, 0], gaze[:, 1]
        x = torch.cos(yaw) * torch.cos(pitch)
        y = torch.sin(yaw) * torch.cos(pitch)
        z = torch.sin(pitch)
        gaze_vector = torch.stack([x, y, z], dim=-1)  # (batch, 3)

        # 获取逆变换矩阵
        H_inv = None

        # 尝试计算逆变换矩阵
        try:
            H_inv = torch.inverse(H)  # 计算逆变换矩阵
        except Exception as e:
            # 如果计算失败，使用伪逆矩阵作为替代
            H_inv = torch.linalg.pinv(H)
            print(f"Warning: Matrix inverse failed. Using pinv instead. Exception: {e}")

        # 如果计算仍然失败，使用单位矩阵
        if H_inv is None:
            H_inv = torch.eye(3, device=H.device).unsqueeze(0).repeat(H.shape[0], 1, 1)

        # 应用逆变换矩阵
        transformed_vector = torch.matmul(H_inv, gaze_vector.unsqueeze(-1))  # (batch, 3, 1)
        transformed_vector = transformed_vector.squeeze(-1)  # (batch, 3)

        # 转换回 (pitch, yaw)
        x, y, z = transformed_vector[..., 0], transformed_vector[..., 1], transformed_vector[..., 2]
        restored_pitch = torch.atan2(z, torch.sqrt(x**2 + y**2))
        restored_yaw = torch.atan2(y, x)

        restored_gaze = torch.stack([restored_pitch, restored_yaw], dim=-1)  # (batch, 2)
        return restored_gaze
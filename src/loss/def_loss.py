import torch
import torch.nn as nn
import numpy as np

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, pred, target):
        # 将预测和目标的 pitch 和 yaw 转换为三维向量
        pred_vector = self.spherical_to_vector(pred)
        target_vector = self.spherical_to_vector(target)
        # 计算预测向量和目标向量之间的余弦相似度
        cosine_similarity = nn.functional.cosine_similarity(pred_vector, target_vector, dim=1)
        # 余弦相似度误差为 1 - cosine_similarity
        loss = 1 - cosine_similarity
        # 返回平均余弦相似度误差
        return torch.mean(loss)

    @staticmethod
    def spherical_to_vector(spherical):
        # 将 pitch 和 yaw 转换为三维向量
        pitch, yaw = spherical[:, 0], spherical[:, 1]
        x = torch.cos(pitch) * torch.cos(yaw)
        y = torch.cos(pitch) * torch.sin(yaw)
        z = torch.sin(pitch)
        return torch.stack([x, y, z], dim=1)
    
def get_loss_in_degree(loss):
    return np.arccos(1 - loss) * 180 / np.pi

class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, pred, target):
        # 将预测和目标的 pitch 和 yaw 转换为三维向量
        pred_vector = self.spherical_to_vector(pred)
        target_vector = self.spherical_to_vector(target)

        # 计算点积
        dot_product = torch.sum(pred_vector * target_vector, dim=1)

        # 裁剪点积值，确保数值稳定性
        dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)  # 添加小的偏移量避免数值问题

        # 计算角度损失
        loss = torch.acos(dot_product)

        # 将弧度转换为度
        loss_in_degrees = loss * (180 / torch.pi)

        # 返回平均角度损失
        return torch.mean(loss_in_degrees)

    @staticmethod
    def spherical_to_vector(spherical):
        # 将 pitch 和 yaw 转换为三维向量
        pitch, yaw = spherical[:, 0], spherical[:, 1]
        x = torch.cos(pitch) * torch.cos(yaw)
        y = torch.cos(pitch) * torch.sin(yaw)
        z = torch.sin(pitch)
        return torch.stack([x, y, z], dim=1)
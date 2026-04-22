import numpy as np

def midpoint(p1, p2):
    """计算两个点之间的中点"""
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

def get_blinking_ratio(eye_points, landmarks):
    """计算眼睛的眨眼比率"""
    # 获取眼睛的左端和右端位置
    left_point = landmarks[eye_points[0]]
    right_point = landmarks[eye_points[3]]
    
    # 获取眼睛的顶部和底部位置
    center_top = landmarks[eye_points[1]]
    center_bottom = landmarks[eye_points[5]]
    
    # 计算眼睛的宽度和高度
    eye_width = right_point[0] - left_point[0]
    eye_height = center_top[1] - center_bottom[1]
    
    # 避免除以零
    if eye_height == 0:
        return 0.0
    
    ratio = eye_width / eye_height
    return ratio

def is_blinking(landmarks):
    """判断眼睛是否闭合"""
    # 定义左右眼的 landmark 索引
    left_eye_points = [36, 37, 38, 39, 40, 41]
    right_eye_points = [42, 43, 44, 45, 46, 47]
    
    # 计算左右眼的眨眼比率
    left_eye_ratio = get_blinking_ratio(left_eye_points, landmarks)
    right_eye_ratio = get_blinking_ratio(right_eye_points, landmarks)
    
    # 计算平均比率
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
    
    # 判断是否闭眼
    return blinking_ratio > 5.7
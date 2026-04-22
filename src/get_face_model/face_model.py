import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(script_dir, '..', '..', 'DFA'))
sys.path.append(os.path.join(script_dir, '..', '..', 'yolov8_face'))

from ultralytics import YOLO
from DFA.TDDFA import TDDFA
from face_detection import detect_face

# 初始化3DDFA模型
tddfa = TDDFA()
yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')

def apply_inverse_transformation(param, vertex):
    """应用逆变换并消除roll和pitch旋转"""
    param_reshaped = param[:12].reshape(3, 4)
    P = param_reshaped[:, :3]
    offset = param_reshaped[:, 3]
    
    # 转置顶点坐标以适配矩阵乘法 (3x68 -> 68x3)
    vertex = vertex.T
    
    # 计算模型坐标系中的顶点
    model_vertex = np.dot(np.linalg.inv(P), (vertex - offset).T).T
    
    # 从姿态参数提取旋转矩阵
    R = P / np.linalg.norm(P, axis=0)  # 去除缩放因子
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    
    # 分解出pitch角度（绕X轴旋转）
    pitch = np.arcsin(-R_clean[2, 0])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), np.sin(pitch)],
                   [0, -np.sin(pitch), np.cos(pitch)]])
    
    # 分解yaw角度（绕Y轴旋转）
    yaw = np.arctan2(R_clean[0, 2], R_clean[2, 2])
    Ry_inv = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])

    # 分解出roll角度（绕Z轴旋转）
    left_eye = np.mean(model_vertex[36:42], axis=0)
    right_eye = np.mean(model_vertex[42:48], axis=0)
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    roll = -np.arctan2(delta_y, delta_x)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    
    # 应用复合旋转
    aligned_vertex = np.dot(Ry_inv, model_vertex.T).T  # 消除yaw
    aligned_vertex = np.dot(Rx, model_vertex.T).T  # 先矫正pitch
    aligned_vertex = np.dot(Rz, aligned_vertex.T).T  # 再矫正roll
    
    return aligned_vertex / 2000.

def rotate_landmarks(landmarks):
    # 假设 landmarks 是一个 (n, 3) 的 NumPy 数组，表示特征点的 (x, y, z) 坐标
    # 计算眼睛的中心点
    left_eye_center = np.mean(landmarks[36:42], axis=0)  # 左眼的特征点索引范围
    right_eye_center = np.mean(landmarks[42:48], axis=0)  # 右眼的特征点索引范围

    # 计算两眼中心的水平和垂直距离
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]

    # 计算旋转角度
    angle = np.arctan2(dY, dX)

    # 计算旋转中心（两眼中心的中点）
    center = ((left_eye_center[0] + right_eye_center[0]) / 2,
              (left_eye_center[1] + right_eye_center[1]) / 2,
              (left_eye_center[2] + right_eye_center[2]) / 2)

    # 创建旋转矩阵（绕Z轴旋转）
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])

    # 应用旋转矩阵到每个特征点
    rotated_landmarks = np.dot(Rz, (landmarks - center).T).T + center

    return rotated_landmarks

def procrustes_analysis(landmarks, reference_landmarks):
    """
    使用Procrustes分析对齐特征点
    :param landmarks: 待对齐的特征点
    :param reference_landmarks: 参考特征点
    :return: 对齐后的特征点
    """
    landmarks_centered = landmarks - np.mean(landmarks, axis=0)
    reference_centered = reference_landmarks - np.mean(reference_landmarks, axis=0)
    U, S, Vt = np.linalg.svd(np.dot(reference_centered.T, landmarks_centered))
    R = np.dot(Vt.T, U.T)
    t = np.mean(reference_landmarks, axis=0) - np.dot(R, np.mean(landmarks, axis=0))
    aligned_landmarks = np.dot(R, landmarks.T).T + t
    return aligned_landmarks

def process_camera(output_txt, max_frames=100):
    cap = cv2.VideoCapture(0)
    frames = []
    frame_count = 0

    # 步骤1：采集视频帧
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Camera', frame)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为RGB格式
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    all_landmarks = []
    for image in frames:
        if image is None:
            continue

        # 使用YOLOv8_face检测人脸框
        detections = detect_face.detect_face_yolo_for_face_model(image, yolo)
        if len(detections) == 0:
            continue

        # 假设只处理第一张人脸
        face_box = [detections[0]]
        # 使用3DDFA获取人脸68个特征点的3D坐标
        param_lst, roi_box_lst = tddfa(image, face_box)  # 批量推理
        vertices = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)  # 获取稀疏顶点（68点）

        # 应用逆变换到每个顶点
        if len(param_lst) > 0 and len(vertices) > 0:
            param = param_lst[0]
            vertex = vertices[0]  # 3x68的顶点坐标
            landmarks = apply_inverse_transformation(param, vertex)
            all_landmarks.append(landmarks)

        # 假设只有一张人脸
        '''landmarks = vertices[0].T
        if landmarks is not None:
            all_landmarks.append(landmarks)'''

    # 假设第50组特征点作为参考
    if len(all_landmarks) > 0:
        reference_landmarks = all_landmarks[49]
        aligned_landmarks = []

        for landmarks in all_landmarks:
            aligned = procrustes_analysis(landmarks, reference_landmarks)
            aligned_landmarks.append(aligned)

        # 去掉不能对齐的数据（例如，对齐误差过大的数据）
        aligned_landmarks = np.array(aligned_landmarks)
        mean_landmarks = np.mean(aligned_landmarks, axis=0)

        # 以鼻尖为原点调整坐标
        nose_tip = mean_landmarks[33]  # 假设鼻尖是第34个点（索引为33）
        mean_landmarks_centered = mean_landmarks - nose_tip

        if os.path.exists(output_txt):
            os.remove(output_txt)
        # 保存到txt文件
        with open(output_txt, "w") as f:
            for point in mean_landmarks_centered:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        print("Average landmarks saved to", output_txt)
    else:
        print("No valid landmarks detected.")

    return mean_landmarks_centered

def process_images(frames, max_frames=50):
    if len(frames) < max_frames:
        max_frames = len(frames)
    frames = frames[:max_frames]

    all_landmarks = []
    for image in frames:
        if image is None:
            continue

        # 使用YOLOv8_face检测人脸框
        detections = detect_face.detect_face_yolo_for_face_model(image, yolo)
        if len(detections) == 0:
            continue

        # 假设只处理第一张人脸
        face_box = [detections[0]]
        # 使用3DDFA获取人脸68个特征点的3D坐标
        param_lst, roi_box_lst = tddfa(image, face_box)  # 批量推理
        vertices = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)  # 获取稀疏顶点（68点）
        # 假设只有一张人脸
        landmarks = vertices[0].T
        if landmarks is not None:
            all_landmarks.append(landmarks)

    # 假设第一组特征点作为参考
    if len(all_landmarks) > 0:
        reference_landmarks = all_landmarks[max_frames//2]
        reference_landmarks = rotate_landmarks(reference_landmarks)
        aligned_landmarks = []

        for landmarks in all_landmarks:
            aligned = procrustes_analysis(landmarks, reference_landmarks)
            aligned_landmarks.append(aligned)

        # 去掉不能对齐的数据（例如，对齐误差过大的数据）
        aligned_landmarks = np.array(aligned_landmarks)
        mean_landmarks = np.mean(aligned_landmarks, axis=0)

        # 以鼻尖为原点调整坐标
        nose_tip = mean_landmarks[33]  # 假设鼻尖是第34个点（索引为33）
        mean_landmarks_centered = mean_landmarks - nose_tip

    return mean_landmarks_centered

def visualize_landmarks(landmarks):
    # 新增可视化辅助线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制特征点
    ax.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2], c='b', marker='o')
    
    # 添加坐标系
    nose_tip = landmarks[33]
    axis_length = 50  # mm
    # X轴 (红色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              axis_length, 0, 0, color='r')
    # Y轴 (绿色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              0, axis_length, 0, color='g')
    # Z轴 (蓝色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              0, 0, axis_length, color='b')
    
    # 设置等比例坐标系
    max_range = np.array([landmarks[:,0].max()-landmarks[:,0].min(),
                          landmarks[:,1].max()-landmarks[:,1].min(),
                          landmarks[:,2].max()-landmarks[:,2].min()]).max() / 2.0
    mid_x = (landmarks[:,0].max()+landmarks[:,0].min()) * 0.5
    mid_y = (landmarks[:,1].max()+landmarks[:,1].min()) * 0.5
    mid_z = (landmarks[:,2].max()+landmarks[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置标题
    ax.set_title('3D Landmarks')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

if __name__ == "__main__":
    output_path = os.path.join(script_dir, '..', 'modules/face_model_test.txt')
    if os.path.exists(output_path):
        os.remove(output_path)
    mean_landmarks_centered = process_camera(output_path, max_frames=100)
    visualize_landmarks(mean_landmarks_centered)
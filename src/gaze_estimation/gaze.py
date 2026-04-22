import torch
import numpy as np
import os
import sys
import cv2
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'face_detection'))

from config_module import config_file
from face_detection import detect_face

config = config_file.config_class()
yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pitch_yaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def pitch_yaw_to_vector_dataset(pitch, yaw):
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    return np.array([x, y, z])

def gaze_prediction(gaze_predictor, face_normalized):
    input_var = face_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
    input_var = trans(input_var)
    input_var = torch.autograd.Variable(input_var.float().cuda())
    input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))
    pred_gaze = gaze_predictor(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[0]  # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = pred_gaze.cpu().data.numpy()
    return pred_gaze_np

def gaze_estimation(gaze_predictor, face_normalized, face_center_3d_origin, rotation_matrix):
    pred_gaze_pitchyaw = gaze_prediction(gaze_predictor, face_normalized) # [pitch, yaw] in arch, with pitch axis pointing upwards, and yaw axis pointing rightwards
    pred_gaze_vector = pitch_yaw_to_vector(np.array([pred_gaze_pitchyaw]))[0]
    pred_gaze_vector_denormalize = np.dot(np.linalg.inv(rotation_matrix), pred_gaze_vector)

    screen_direction = config.screen.screen_direction
    screen_normal = screen_direction / np.linalg.norm(screen_direction)
    ray_origin = face_center_3d_origin
    ray_direction = pred_gaze_vector_denormalize

    return ray_origin, ray_direction, screen_normal

def gaze_estimation_dataset(gaze_predictor, face_normalized, rotation_matrix):
    pred_gaze_pitchyaw = gaze_prediction(gaze_predictor, face_normalized) # [pitch, yaw] in arch, with pitch axis pointing upwards, and yaw axis pointing rightwards
    pred_gaze_vector = pitch_yaw_to_vector_dataset(pred_gaze_pitchyaw[0], pred_gaze_pitchyaw[1])
    pred_gaze_vector_denormalize = np.dot(np.linalg.inv(rotation_matrix), pred_gaze_vector)
    ray_direction = pred_gaze_vector_denormalize

    return ray_direction

def gaze_prediction_v2(gaze_predictor, frame):
    # 转换为CHW格式并应用预处理
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)  # 转换为PIL Image
    input_var = val_transform(frame)
    input_var = torch.autograd.Variable(input_var.float().cuda())
    input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))
    pred_gaze = gaze_predictor(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[0]  # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = pred_gaze.cpu().data.numpy()
    return pred_gaze_np

def gaze_estimation_v2(gaze_predictor, frame, face_center_3d_origin):
    pred_gaze_pitchyaw = gaze_prediction_v2(gaze_predictor, frame) # [pitch, yaw] in arch, with pitch axis pointing upwards, and yaw axis pointing rightwards
    pred_gaze_vector = pitch_yaw_to_vector(np.array([pred_gaze_pitchyaw]))[0]
    ray_direction = pred_gaze_vector

    screen_direction = config.screen.screen_direction
    screen_normal = screen_direction / np.linalg.norm(screen_direction)
    ray_origin = face_center_3d_origin

    return ray_origin, ray_direction, screen_normal

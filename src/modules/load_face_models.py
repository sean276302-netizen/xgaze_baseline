import dlib
import os
import sys
import numpy as np
import face_alignment

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

# from get_face_model.face_model import process_camera

def load_face_models():
    face_landmark_predictor = dlib.shape_predictor(os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat"))
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    # if not os.path.exists(os.path.join(script_dir, "face_model_actual.txt")):
    #     process_camera(os.path.join(script_dir, "face_model_actual.txt"), max_frames=100)
    face_model = np.loadtxt(os.path.join(script_dir,'face_model.txt'))
    face_model_std = np.loadtxt(os.path.join(script_dir,'face_model.txt'))
    face_model_ratio = calculate_face_size_ratio(face_model, face_model_std)
    if face_model.shape[0] != 68: face_model_ratio = 1.0
    face_alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    return face_landmark_predictor, face_detector, face_model, face_alignment_model, face_model_ratio

def load_face_models_v2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_landmark_predictor = dlib.shape_predictor(os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat"))
    if not os.path.exists(os.path.join(script_dir, "face_model_actual.txt")):
        process_camera(os.path.join(script_dir, "face_model_actual.txt"), max_frames=100)
    face_model = np.loadtxt(os.path.join(script_dir,'face_model_actual.txt'))
    return face_landmark_predictor, face_model

def calculate_face_size_ratio(face1, face2):
    # 提取关键特征点的3D坐标
    landmarks_1 = np.array(face1)
    landmarks_2 = np.array(face2)

    # 定义关键特征点索引
    left_inner_corner_1 = landmarks_1[36]
    left_outer_corner_1 = landmarks_1[39]
    right_inner_corner_1 = landmarks_1[42]
    right_outer_corner_1 = landmarks_1[45]
    left_nose_wing_1 = landmarks_1[30]
    right_nose_wing_1 = landmarks_1[31]

    left_inner_corner_2 = landmarks_2[20]
    left_outer_corner_2 = landmarks_2[23]
    right_inner_corner_2 = landmarks_2[26]
    right_outer_corner_2 = landmarks_2[29]
    left_nose_wing_2 = landmarks_2[15]
    right_nose_wing_2 = landmarks_2[19]

    # 计算特征长度
    eye_width_1 = np.linalg.norm(left_outer_corner_1 - left_inner_corner_1)
    eye_width_2 = np.linalg.norm(right_outer_corner_1 - right_inner_corner_1)
    eye_spacing_1 = np.linalg.norm(left_inner_corner_1 - right_inner_corner_1)
    nose_width_1 = np.linalg.norm(left_nose_wing_1 - right_nose_wing_1)

    eye_width_3 = np.linalg.norm(left_outer_corner_2 - left_inner_corner_2)
    eye_width_4 = np.linalg.norm(right_outer_corner_2 - right_inner_corner_2)
    eye_spacing_2 = np.linalg.norm(left_inner_corner_2 - right_inner_corner_2)
    nose_width_2 = np.linalg.norm(left_nose_wing_2 - right_nose_wing_2)

    # 计算比例
    eye_width_ratio = (eye_width_1 + eye_width_2) / (eye_width_3 + eye_width_4)
    eye_spacing_ratio = eye_spacing_1 / eye_spacing_2
    nose_width_ratio = nose_width_1 / nose_width_2

    # 综合比例
    overall_ratio = (eye_width_ratio + eye_spacing_ratio + nose_width_ratio) / 3

    return overall_ratio
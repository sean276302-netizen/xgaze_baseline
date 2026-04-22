import os
import cv2
import sys
import numpy as np
from imutils import face_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from blink import blink
from config_module import config_file

config = config_file.config_class()

def estimate_head_pose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def crop_head(image, detections, output_size=(448, 448), padding_ratio=config.train.face_extending_ratio):
    if image is None:
        raise ValueError("无法加载图像，请检查路径是否正确")

    # 选择置信度最高的检测结果
    best_detection = detections[0]

    # 解析检测结果
    x1, y1, x2, y2 = best_detection
    width = x2 - x1
    height = y2 - y1

    # 增加边界范围
    delta_w = width * padding_ratio
    delta_h = height * padding_ratio

    # 调整边界
    x1_new = int(max(0, x1 - delta_w))
    y1_new = int(max(0, y1 - delta_h))
    x2_new = int(min(image.shape[1], x2 + delta_w))
    y2_new = int(min(image.shape[0], y2 + delta_h))

    # 裁切图像
    head_image = image[y1_new:y2_new, x1_new:x2_new]

    # 缩放和填充逻辑
    target_h, target_w = output_size
    current_h, current_w = head_image.shape[:2]

    # 确定最长边方向
    if current_w > current_h:
        scale_ratio = target_w / current_w
        new_h = int(current_h * scale_ratio)
        new_w = target_w
    else:
        scale_ratio = target_h / current_h
        new_h = target_h
        new_w = int(current_w * scale_ratio)

    # 缩放图像
    resized_image = cv2.resize(head_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算需要填充的边框大小
    pad_height = target_h - new_h
    pad_width = target_w - new_w

    # 根据最长边方向填充
    if new_w == target_w:  # 原图宽比高长，缩放后宽达到目标宽，填充上下
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = 0
        pad_right = 0
    else:  # 原图高比宽长，缩放后高达到目标高，填充左右
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = 0
        pad_bottom = 0

    # 用黑色填充
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

def normalize_data_face(img, face_model, landmarks, hr, ht, cam, face_model_ratio):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600 * face_model_ratio   # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    EAR = []
    for i in range(2):
        EAR.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
            landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / (
                               2 * np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i], 2)))
    EAR = np.mean(np.asarray(EAR))

    return img_warped, R, face_center, EAR

def normalization(detected_faces, frame, face_landmark_predictor, face_3D_generic_model, camera_matrix, camera_distortion, face_model_ratio):
    shape = face_landmark_predictor(frame, detected_faces[0])
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)
    
    is_blinking = blink.is_blinking(landmarks)

    ## estimate head pose
    landmark_use = [20, 23, 26, 29, 15, 19]  ## we use eye corners and nose conners
    if face_3D_generic_model.shape[0] == 68: landmark_use = [36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59]
    face_model = face_3D_generic_model[landmark_use, :]
    facePts = face_model.reshape(len(landmark_use), 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59], :]
    if face_3D_generic_model.shape[0] == 50: landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(len(landmark_use), 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimate_head_pose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    face_normalized, rotation_matrix, face_center_3d_origin, EAR = normalize_data_face(frame, face_model, landmarks, hr, ht, camera_matrix, face_model_ratio)

    face_center_3d_origin = -face_center_3d_origin
    face_center_3d_origin = face_center_3d_origin.T[0]

    return face_normalized, face_center_3d_origin, rotation_matrix, EAR, is_blinking

def normalization_v2(detected_faces, detected_faces_for_crop_head, frame, face_landmark_predictor, face_3D_generic_model, camera_matrix, camera_distortion):
    shape = face_landmark_predictor(frame, detected_faces[0])
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)
    
    is_blinking = blink.is_blinking(landmarks)

    ## estimate head pose
    landmark_use = [20, 23, 26, 29, 15, 19]  ## we use eye corners and nose conners
    if face_3D_generic_model.shape[0] == 68: landmark_use = [36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59]
    face_model = face_3D_generic_model[landmark_use, :]
    facePts = face_model.reshape(len(landmark_use), 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59], :]
    #landmarks_sub = landmarks[[i for i in range(68)], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(len(landmark_use), 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimate_head_pose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center_3d_origin = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    face_center_3d_origin = -face_center_3d_origin
    face_center_3d_origin = face_center_3d_origin.T[0]

    EAR = []
    for i in range(2):
        EAR.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
            landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / (
                               2 * np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i], 2)))
    EAR = np.mean(np.asarray(EAR))

    head_image = crop_head(frame, detected_faces_for_crop_head)

    return head_image, face_center_3d_origin, EAR, is_blinking

def normalizeData(img, face_model, hr, ht, cam, face_model_ratio):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 500 * face_model_ratio   # normalized distance between eye and camera
    roiSize = (128, 128)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    #gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of left eye
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of right eye

    ## normalize each eye
    data = []
    for et in [re, le]:
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et)  # actual distance between eye and original camera

        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        ## ---------- normalize gaze vector ----------
        #gc_normalized = gc - et  # gaze vector
        #gc_normalized = np.dot(R, gc_normalized)
        #gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        data.append([img_warped, hr_norm, R])

    return data[0][0]

def normalization_hw(detected_faces, frame, face_landmark_predictor, face_3D_generic_model, camera_matrix, camera_distortion, face_model_ratio):
    shape = face_landmark_predictor(frame, detected_faces[0])
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)
    
    is_blinking = blink.is_blinking(landmarks)

    ## estimate head pose
    landmark_use = [20, 23, 26, 29, 15, 19]  ## we use eye corners and nose conners
    if face_3D_generic_model.shape[0] == 68: landmark_use = [36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59]
    face_model = face_3D_generic_model[landmark_use, :]
    facePts = face_model.reshape(len(landmark_use), 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35, 48, 54, 30, 8, 4, 12, 37, 41, 46, 50, 57, 62, 51, 59], :]
    if face_3D_generic_model.shape[0] == 50: landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(len(landmark_use), 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimate_head_pose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    face_normalized = normalizeData(frame, face_model, hr, ht, camera_matrix, face_model_ratio)

    return face_normalized
import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))
sys.path.append(os.path.join(script_dir, '..'))

from config_module import config_file
from calibration import calibration

config = config_file.config_class()

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def fig_to_np_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()  # Render the figure
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(height, width, 3)
    return img

def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
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

    '''Ear = []
    for i in range(2):
        Ear.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
            landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / (
                               2 * np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i], 2)))
    Ear = np.mean(np.asarray(Ear))'''

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    #landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    #landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, R, face_center

def normalizeData(img, face_model, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
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


if __name__ == '__main__':
    path_to_shape_predictor = os.path.join(script_dir, "../modules/shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(path_to_shape_predictor)
    # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful

    # load camera information
    path_to_calibration = os.path.join(script_dir, "../../test_hw/calibration/mono.xml")
    cam_file_name = path_to_calibration  # this is camera calibration information file obtained with OpenCV
    if not os.path.isfile(cam_file_name):
        print('no camera calibration file is found.')
        exit(0)
    print(cam_file_name)
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat() # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('dist_coeffs').mat()

    image_path = "E:/hw_test_dataset_B/p75/p75_084932_0000.jpg"
    image = cv2.imread(image_path)

    detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face')
        exit(0)
    print('detected one face')
    shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)

    print('estimate head pose')
    # load face model
    face_model_load = np.loadtxt(os.path.join(script_dir,'../modules/face_model.txt'))  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    print('data normalization, i.e. crop the face image')
    data = normalizeData(image, face_model,  hr, ht, camera_matrix)
    
    print('prepare the output')
    
    for i in range(2):
        # 指定输出文件夹路径
        output_folder = "test_hw/normalized/p75"

        # 如果文件夹不存在，则创建它
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 指定输出路径
        output_path = os.path.join(output_folder, 'output%d.jpg' % i)

        cv2.imwrite(output_path, data[i][0])
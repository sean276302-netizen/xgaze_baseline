import sys
import os
import cv2
import dlib
import time
import numpy as np
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))

from config_module import config_file
from normalization import normalization
from intersection import intersection
from interaction import cursor_control
from modules import load_face_models
from gaze_estimation import gaze
from calibration import calibration
from src.model.init_model import init
from visualization import visualization
from face_detection import detect_face

config = config_file.config_class()


if __name__ == '__main__':
    ## load screen and camera data
    w_screen = config.screen.w_screen
    h_screen = config.screen.h_screen
    screen_width = config.screen.screen_width
    screen_height = config.screen.screen_height
    screen_center = config.screen.screen_center
    pixel_scale=config.screen.pixel_scale
    camera_position = config.camera.camera_position

    ## load camera data
    camera_matrix, camera_distortion = calibration.load_calibration()

    ## load face models
    face_landmark_predictor, face_detector, face_3D_generic_model, face_alignment_model, face_model_ratio = load_face_models.load_face_models()

    face_3D_generic_model /= face_model_ratio
    face_3D_generic_model = np.loadtxt(os.path.join(script_dir, 'src/modules/face_model.txt'))
    face_model_ratio = 1.0

    ## init model
    gaze_predictor = init()
    yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')

    ## set cv2 font
    font = cv2.FONT_HERSHEY_PLAIN

    ## capture image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度

    if not cap.isOpened():
        print("fail to open camera!")
        cap.release()
        exit()

    no_face_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("fail to read frame!")
            continue
        if no_face_counter > 300:
            print("no face detected for 30 frames, exit.")
            break
        ## detect face
        p1 = time.time()
        detected_faces = detect_face.detect_face_yolo(frame, yolo)
        if len(detected_faces) == 0:
            print("No faces detected in the frame.")
            no_face_counter += 1
            continue
        print("detect one face.")

        ## data normalization method
        face_normalized, face_center_3d_origin, rotation_matrix, EAR, is_blinking = (
            normalization.normalization(detected_faces, frame, 
                                        face_landmark_predictor, 
                                        face_3D_generic_model, 
                                        camera_matrix,
                                        camera_distortion,
                                        face_model_ratio))
        if face_normalized is None:
            print('normalization failed')
            continue

        ## estimate the gaze vector
        ray_origin, ray_direction, screen_normal = (
            gaze.gaze_estimation(gaze_predictor, face_normalized, 
                                    face_center_3d_origin, rotation_matrix))

        ## calculate the intersection point of the ray and the screen plane
        intersection_point = (
            intersection.ray_plane_intersection(ray_origin, ray_direction, 
                                                screen_center, screen_normal))
        if intersection_point is None:
            print('intersection failed')
            continue
        
        ## calculate the PoG of the intersection point on the screen
        PoG_x_pred, PoG_y_pred = (
            intersection.point_to_screen_coordinate(intersection_point, 
                                                    screen_center, screen_normal, 
                                                    pixel_scale, w_screen, h_screen))

        ## control the mouse cursor
        try:
            cursor_control.control_mouse(PoG_x_pred, PoG_y_pred, w_screen, h_screen)
        except:
            print('cursor control failed')

        ## visualization
        ## if EAR is -1, it means that the face is too small to be detected
        if (EAR == -1):
            continue

        ## draw the gaze vector and the intersection point on the screen
        '''visualization.visualization(frame, face_normalized, is_blinking, camera_position, 
                                    screen_center, screen_normal, screen_width, 
                                    screen_height, ray_origin, ray_direction, 
                                    intersection_point, font)'''
        p2 = time.time()
        print('time cost:', p2-p1)

        key = cv2.waitKey(1)
        # close the webcam when escape key is pressed
        if key == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
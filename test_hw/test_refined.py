import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))
sys.path.append(os.path.join(script_dir, '..'))

from config_module import config_file
from normalization import normalization
from intersection import intersection
from interaction import cursor_control
from modules import load_face_models
from gaze_estimation import gaze
from calibration import calibration
from model import init_model
from visualization import visualization
from face_detection import detect_face
from get_face_model import face_model
from refinement import refinement

config = config_file.config_class()

def get_history(image_path, POG_HISTORY, BLINK, txt_path, face_3D_generic_model, face_model_ratio, camera_matrix, camera_distortion):
    frame = cv2.imread(image_path)

    ## detect face
    detected_faces = detect_face.detect_face_yolo(frame, yolo)
    if len(detected_faces) == 0:
        print("No faces detected in the frame: ", image_path)
        no_face_counter += 1
        return None

    ## data normalization method
    face_normalized, face_center_3d_origin, rotation_matrix, EAR, is_blinking = (
        normalization.normalization(detected_faces, frame, 
                                    face_landmark_predictor, 
                                    face_3D_generic_model, 
                                    camera_matrix,
                                    camera_distortion,
                                    face_model_ratio))
    if face_normalized is None:
        print('normalization failed: ', image_path)
        return None
    
    '''save_path = os.path.join(f"test_hw/normalized", os.path.basename(image_path))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, face_normalized)'''

    ## estimate the gaze vector
    ray_origin, ray_direction, screen_normal = (
        gaze.gaze_estimation(gaze_predictor, face_normalized, 
                                face_center_3d_origin, rotation_matrix))

    ## calculate the intersection point of the ray and the screen plane
    intersection_point = (
        intersection.ray_plane_intersection(ray_origin, ray_direction, 
                                            screen_center, screen_normal))
    if intersection_point is None:
        print('intersection failed: ', image_path)
        return None
    
    ## calculate the PoG of the intersection point on the screen
    PoG_x_pred, PoG_y_pred = (
        intersection.point_to_screen_coordinate(intersection_point, 
                                                screen_center, screen_normal, 
                                                pixel_scale, w_screen, h_screen))


    POG_HISTORY = POG_HISTORY + [(PoG_x_pred, PoG_y_pred), ]
    BLINK = BLINK + [is_blinking, ]    #True -> blinking

    return POG_HISTORY, BLINK

def main(image_path, POG_HISTORY, BLINK, txt_path, face_3D_generic_model, face_model_ratio, camera_matrix, camera_distortion):
    frame = cv2.imread(image_path)

    ## detect face
    detected_faces = detect_face.detect_face_yolo(frame, yolo)
    if len(detected_faces) == 0:
        print("No faces detected in the frame: ", image_path)
        no_face_counter += 1
        return None

    ## data normalization method
    face_normalized, face_center_3d_origin, rotation_matrix, EAR, is_blinking = (
        normalization.normalization(detected_faces, frame, 
                                    face_landmark_predictor, 
                                    face_3D_generic_model, 
                                    camera_matrix,
                                    camera_distortion,
                                    face_model_ratio))
    if face_normalized is None:
        print('normalization failed: ', image_path)
        return None
    
    '''save_path = os.path.join(f"test_hw/normalized", os.path.basename(image_path))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, face_normalized)'''

    ## estimate the gaze vector
    ray_origin, ray_direction, screen_normal = (
        gaze.gaze_estimation(gaze_predictor, face_normalized, 
                                face_center_3d_origin, rotation_matrix))

    ## calculate the intersection point of the ray and the screen plane
    intersection_point = (
        intersection.ray_plane_intersection(ray_origin, ray_direction, 
                                            screen_center, screen_normal))
    if intersection_point is None:
        print('intersection failed: ', image_path)
        return None
    
    ## calculate the PoG of the intersection point on the screen
    PoG_x_pred, PoG_y_pred = (
        intersection.point_to_screen_coordinate(intersection_point, 
                                                screen_center, screen_normal, 
                                                pixel_scale, w_screen, h_screen))


    ## visualization
    ## if EAR is -1, it means that the face is too small to be detected
    if (EAR == -1):
        print("EAR is -1, the face is too small to be detected: ", image_path)
        return None
    else:
        # PoG_x_pred, PoG_y_pred, offset = refinement.refine_x(PoG_x_pred, PoG_y_pred, POG_HISTORY, BLINK, w_screen, h_screen)

        with open(txt_path, 'a') as f:
            f.write(str(PoG_x_pred) + ',' + str(PoG_y_pred) + '\n')

if __name__ == '__main__':
    ## load screen and camera data
    w_screen = config.hw_test_screen.w_screen
    h_screen = config.hw_test_screen.h_screen
    screen_center = config.hw_test_screen.screen_center
    pixel_scale=config.hw_test_screen.pixel_scale
    camera_position = config.camera.camera_position

    ## load face models
    face_landmark_predictor, face_detector, face_3D_generic_model, face_alignment_model, face_model_ratio = load_face_models.load_face_models()

    ## init model
    gaze_predictor = init_model.init()
    yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')


    for camera in ["mono"]:
        if camera == "rgb":
            root_dir = "E:/hw_test_dataset_A"
        else:
            root_dir = "E:/hw_test_dataset_B"

        camera_matrix, camera_distortion = calibration.load_calibration_from_file(f"test_hw/calibration/{camera}.xml")

        for participat in os.listdir(root_dir):
            for pose in os.listdir(os.path.join(root_dir, participat)):
                POG_HISTORY = []
                BLINK = []
                txt_path = os.path.join(f"test_hw/pred_PoG_refined/{camera}/{participat}", pose + ".txt")
                if not os.path.exists(os.path.dirname(txt_path)):
                    os.makedirs(os.path.dirname(txt_path))
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                images = []
                image_paths = []
                pose_dir = os.path.join(root_dir, participat, pose)
                if os.path.isdir(pose_dir):
                    for image_name in os.listdir(pose_dir):
                        if image_name.endswith('.jpg'):
                            image_path = os.path.join(pose_dir, image_name)
                            image_paths.append(image_path)
                            image = cv2.imread(image_path)
                            if image is not None:
                                images.append(image)
                            else:
                                print("Failed to load image: ", image_path)

                # face_3D_generic_model = face_model.process_images(images, max_frames=50)
                face_model_std = np.loadtxt(os.path.join(script_dir,'..', 'src/modules/face_model.txt'))
                # face_model_ratio = load_face_models.calculate_face_size_ratio(face_3D_generic_model, face_model_std)
                # face_3D_generic_model /= face_model_ratio
                face_model_ratio = 1.0

                progress_bar = tqdm(image_paths, desc=f"Processing {camera} images for {participat}", unit="image")
                for image_path in progress_bar:
                    POG_HISTORY, BLINK = get_history(image_path, POG_HISTORY, BLINK, txt_path, face_3D_generic_model, face_model_ratio, camera_matrix, camera_distortion)
                
                for image_path in progress_bar:
                    main(image_path, POG_HISTORY, BLINK, txt_path, face_model_std, face_model_ratio, camera_matrix, camera_distortion)
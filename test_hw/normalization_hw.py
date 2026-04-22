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

config = config_file.config_class()

def main(image_path, face_3D_generic_model, face_model_ratio, camera_matrix, camera_distortion, camera):
    frame = cv2.imread(image_path)

    ## detect face
    detected_faces = detect_face.detect_face_yolo(frame, yolo)
    if len(detected_faces) == 0:
        print("No faces detected in the frame: ", image_path)
        no_face_counter += 1
        return None

    ## data normalization method
    face_normalized = (
        normalization.normalization_hw(detected_faces, frame, 
                                    face_landmark_predictor, 
                                    face_3D_generic_model, 
                                    camera_matrix,
                                    camera_distortion,
                                    face_model_ratio))
    if face_normalized is None:
        print('normalization failed: ', image_path)
        return None
    
    # 将图片放大到600*600
    face_normalized = cv2.resize(face_normalized, (600, 600))

    image_name = os.path.join(camera, image_path.split("\\")[-2], image_path.split("\\")[-1])
    save_path = os.path.join("D:/users/annual_project_of_grade1/normalized_hw", image_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, face_normalized)



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


    for camera in ["rgb", "mono"]:
        if camera == "rgb":
            root_dir = "E:/hw_test_dataset_A"
        else:
            root_dir = "E:/hw_test_dataset_B"

        camera_matrix, camera_distortion = calibration.load_calibration_from_file(f"test_hw/calibration/{camera}.xml")

        for participat in os.listdir(root_dir):
            images = []
            image_paths = []
            participation_dir = os.path.join(root_dir, participat)
            if os.path.isdir(participation_dir):
                for image_name in os.listdir(participation_dir):
                    if image_name.endswith('.jpg'):
                        image_path = os.path.join(participation_dir, image_name)
                        image_paths.append(image_path)
                        image = cv2.imread(image_path)
                        if image is not None:
                            images.append(image)
                        else:
                            print("Failed to load image: ", image_path)

            face_3D_generic_model = face_model.process_images(images, max_frames=50)
            face_model_std = np.loadtxt(os.path.join(script_dir,'..', 'src/modules/face_model.txt'))
            face_model_ratio = load_face_models.calculate_face_size_ratio(face_3D_generic_model, face_model_std)
            face_3D_generic_model /= face_model_ratio
            face_model_ratio = 1.0

            progress_bar = tqdm(image_paths, desc=f"Processing {camera} images for {participat}", unit="image")
            for image_path in progress_bar:
                main(image_path, face_model_std, face_model_ratio, camera_matrix, camera_distortion, camera)
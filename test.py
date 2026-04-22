import sys
import os
import cv2
import time
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))
sys.path.append(os.path.join(script_dir, 'yolov8_face'))

from config_module import config_file
from normalization import normalization
from intersection import intersection
from modules import load_face_models
from gaze_estimation import gaze
from calibration import calibration
from model import init_model
from face_detection import detect_face

config = config_file.config_class()

if __name__ == '__main__':
    i = int(input("请输入图片编号："))
    txt_path = "src/test_module/gaze/%d.txt" % i
    if os.path.exists(txt_path):
        os.remove(txt_path)

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
    (face_landmark_predictor, face_detector, 
     face_3D_generic_model, face_alignment_model, face_model_ratio) = (
        load_face_models.load_face_models())

    ## init model
    gaze_predictor = init_model.init()
    yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')

    ## set cv2 font
    font = cv2.FONT_HERSHEY_PLAIN

    ## capture image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度

    # 读取图片
    image_path = "D:/users/annual_project_of_grade1/test_images/%d.jpg" % i
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片：{image_path}")

    # 创建窗口并设置为全屏
    window_name = "Full Screen Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # 设置为全屏

    # 显示图片
    cv2.imshow(window_name, image)

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
        p1 = time.time()
        if no_face_counter > 30:
            print("no face detected for 30 frames, exit.")
            break
        ## detect face
        detected_faces = detect_face.detect_face_yolo(frame, yolo)
        if len(detected_faces) == 0:
            print("\nNo faces detected in the frame.")
            no_face_counter += 1
            continue
        print("\ndetect one face.")
        p2 = time.time()

        ## data normalization method
        face_normalized, face_center_3d_origin, rotation_matrix, EAR, is_blinking = (
            normalization.normalization(detected_faces, frame, 
                                        face_landmark_predictor, 
                                        face_3D_generic_model, 
                                        camera_matrix, camera_distortion, face_model_ratio))
        if face_normalized is None:
            print('normalization failed')
            continue
        p3 = time.time()

        ## estimate the gaze vector
        ray_origin, ray_direction, screen_normal = (
            gaze.gaze_estimation(gaze_predictor, face_normalized, 
                                    face_center_3d_origin, rotation_matrix))
        p4 = time.time()

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
        p5 = time.time()
        print("total time:", p5-p1, '\n')
        with open(txt_path, 'a') as f:
            f.write(str(PoG_x_pred) + ',' + str(PoG_y_pred) + '\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
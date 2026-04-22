import os
import sys
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(script_dir, '..', '..'))

from config_module import config_file
from ultralytics import YOLO
from face_detection import detect_face

yolo = YOLO('yolov8_face/weights/yolov8n-face.pt')
config = config_file.config_class()

def crop_head(image, output_size=(448, 448), padding_ratio=config.train.face_extending_ratio):
    if image is None:
        raise ValueError("无法加载图像，请检查路径是否正确")

    detections = detect_face.detect_face_yolo_for_face_model(image, yolo)

    # 如果没有检测到人脸，返回 None
    if len(detections) == 0:
        return None

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

if __name__ == '__main__':
    output_size = (448, 448)
    for i in range(9):
        image_path = "D:\\users\\annual_project_of_grade1\\pictures\\%d.jpg" % i
        image = cv2.imread(image_path)
        head_image = crop_head(image, output_size=output_size, padding_ratio=0.5)
        cv2.imwrite("D:\\users\\annual_project_of_grade1\\pictures\\output_v3_%d.jpg" % i, head_image)
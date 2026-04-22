import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from config_module import config_file

config = config_file.config_class()

def create_image(i):
    output_path = 'D:/users/annual_project_of_grade1/test_images/%d.jpg' % i
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if os.path.exists(output_path):
        width, height = config.screen.w_screen, config.screen.h_screen  # resolution
        w_d = (width - 50*2) // 4
        h_d = (height - 50*2) // 3
        line = i // 5
        row = i % 5
        x = 50 + w_d * row
        y = 50 + h_d * line

        return x, y
    else:
        width, height = config.screen.w_screen, config.screen.h_screen  # 分辨率
        w_d = (width - 50*2) // 4
        h_d = (height - 50*2) // 3
        line = i // 5
        row = i % 5
        x = 50 + w_d * row
        y = 50 + h_d * line
        black_dot_position = (x, y)  # 黑点的位置（x, y）
        black_dot_size = 10  # 黑点的大小（半径）

        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.circle(frame, black_dot_position, black_dot_size, (0, 0, 0), -1)
        cv2.imwrite(output_path, frame)

        if os.path.exists(output_path):
            print(f"图片生成成功：{output_path}")
        else:
            print(f"图片生成失败：{output_path}")

        return x, y
    
for i in range(20):
    x, y = create_image(i)
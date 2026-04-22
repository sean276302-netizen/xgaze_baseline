import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Rectangle

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config_module import config_file

config = config_file.config_class()

# 屏幕规格
SCREEN_WIDTH, SCREEN_HEIGHT = config.hw_test_screen.w_screen, config.hw_test_screen.h_screen
pixel_scale = config.hw_test_screen.pixel_scale

# 读取预测点数据
def read_points_from_file(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            x, y = map(float, line.strip().split(","))
            points.append((x, y))
    return np.array(points)

# 绘制预测点和注视点
def plot_points(points, screen_width, screen_height, show_number=False, participant_id=None, savefig_path=None):
    phone_width = 1216 * 0.0583 / config.hw_test_screen.pixel_scale
    phone_height = 2688 * 0.0583 / config.hw_test_screen.pixel_scale

    # 创建图形
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 设置背景为白色
    ax.set_facecolor('white')

    # 绘制屏幕边框（矩形）
    screen_border = Rectangle((0, 0), screen_width, screen_height, 
                              edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(screen_border)

    # 绘制预测点（蓝色点）
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=20, label='Predicted Points')

    if show_number:
        for i, (x, y) in enumerate(points):
            plt.text(x, y, f'{i}', color='red', fontsize=10, ha='right')  # 在点旁边标注序号

    camera_position = (
        screen_width / 2, 
        10.43 / config.hw_test_screen.pixel_scale
        )
    phone_border = Rectangle((camera_position[0] - phone_width / 2, camera_position[1] - (77.9 * 0.0583 / config.hw_test_screen.pixel_scale)), 
                             phone_width, phone_height, 
                             edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(phone_border)

    # 添加摄像头位置
    camera_x = camera_position[0]
    camera_y = camera_position[1]
    plt.plot(camera_x, camera_y, 'ro', markersize=8, label='Camera Position')  # 红色点表示摄像头位置


    # 设置坐标轴范围
    plt.xlim(-screen_width * 0., screen_width * 1.)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0., screen_height * 1.)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.25))  # 将图例放置在图像右侧

    # 设置标题和坐标轴标签
    plt.title(f'p{participant_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.savefig(savefig_path, dpi=300)

    # 显示图像
    #plt.show()

if __name__ == '__main__':
    for mode in ['rgb', 'mono']:
        if mode == 'rgb':
            participant_id_list = [6, 8, 9, 10, 11]
        elif mode == 'mono':
            participant_id_list = [75, 76, 77, 78]
        for participant_id in participant_id_list:
            txt_path = f"test_hw/pred_PoG/{mode}/p{participant_id:02d}.txt"

            points = read_points_from_file(txt_path)

            participant_id = txt_path.split("/")[-1].split(".")[0][1:]

            save_path = f'D:/users/annual_project_of_grade1/hw_test/pred_PoG/{txt_path.split("/")[-2]}/{txt_path.split("/")[-1]}'

            savefig_path = f'D:/users/annual_project_of_grade1/hw_test/result/{txt_path.split("/")[-2]}/{txt_path.split("/")[-1].split(".")[0]}.png'

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            if not os.path.exists(os.path.dirname(savefig_path)):
                os.makedirs(os.path.dirname(savefig_path))

            #np.savetxt(save_path, points, delimiter=',')

            # 绘制预测点
            plot_points(points, 
                        SCREEN_WIDTH, 
                        SCREEN_HEIGHT, 
                        show_number=False, 
                        participant_id=participant_id,
                        savefig_path=savefig_path)
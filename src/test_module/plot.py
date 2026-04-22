import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from config_module import config_file
from create_image import create_image

config = config_file.config_class()

# 屏幕规格
SCREEN_WIDTH, SCREEN_HEIGHT = config.screen.w_screen, config.screen.h_screen
pixel_scale = config.screen.pixel_scale

# 读取预测点数据
def read_points_from_file(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            x, y = map(float, line.strip().split(","))
            points.append((x, y))
    return np.array(points)

def calculate_distances(points, gaze_point):
    distances = np.sqrt((points[:, 0] - gaze_point[0]) ** 2 + (points[:, 1] - gaze_point[1]) ** 2)
    return distances

# 绘制预测点和注视点
def plot_points(points, gaze_point, screen_width, screen_height, distances):
    # 创建图形
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 设置背景为白色
    ax.set_facecolor('white')

    # 绘制屏幕边界
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
    plt.axvline(x=screen_width, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=screen_height, color='black', linestyle='-', linewidth=2)

    # 绘制预测点（蓝色点）
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=20, label='Predicted Points')
    # 计算平均预测点
    mean_point = np.mean(points, axis=0)
    plt.plot(mean_point[0], mean_point[1], 'go', markersize=10, label='Mean Predicted Point')  # 绿色点表示平均预测点

    # 绘制真实注视点（红色点）
    plt.plot(gaze_point[0], gaze_point[1], 'ro', markersize=10, label='Gaze Point')

    # 设置坐标轴范围
    plt.xlim(-screen_width * 0.1, screen_width * 1.1)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0.1, screen_height * 1.1)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.25))  # 将图例放置在图像右侧

    # 设置标题和坐标轴标签
    plt.title('Predicted Points with Gaze Point')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 计算平均距离和标准差
    mean_distance = np.mean(distances) * pixel_scale  # 单位为mm
    std_distance = np.std(distances) * pixel_scale  # 单位为mm

    # 将统计量注释显示在图像右侧
    plt.text(screen_width * 1.02, screen_height * 0.9, 
             f"Mean Distance: {mean_distance:.2f}mm\nStandard Deviation: {std_distance:.2f}mm",
             fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, verticalalignment='top')

    # 显示图像
    plt.show()

i = int(input("请输入测试图片编号："))  # 示例图片编号，根据实际情况调整

file_path = "D:/users/annual_project_of_grade1/gaze_estimation/src/test_module/gaze/%d.txt" % i  # 替换为你的预测点文件路径

if not os.path.exists(file_path):
    print("\n预测点文件不存在！")
    exit()

gaze_point = create_image(i)  # 示例真实注视点，根据实际情况调整

print(f"注视点坐标：{gaze_point}")

# 读取预测点
points = read_points_from_file(file_path)

# 计算距离
distances = calculate_distances(points, gaze_point)

# 绘制预测点和注视点，并添加统计量注释
plot_points(points, gaze_point, SCREEN_WIDTH, SCREEN_HEIGHT, distances)
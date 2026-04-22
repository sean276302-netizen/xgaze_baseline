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
def plot_points(points, screen_width, screen_height):
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

    # 设置坐标轴范围
    plt.xlim(-screen_width * 0.1, screen_width * 1.1)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0.1, screen_height * 1.1)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.25))  # 将图例放置在图像右侧

    # 设置标题和坐标轴标签
    plt.title('Predicted Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 显示图像
    plt.show()

def plot_points_hw(points, screen_width, screen_height):
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

    # 在每个点旁边标注序号
    '''for i, (x, y) in enumerate(points):
        plt.text(x, y, f'{i}', color='red', fontsize=10, ha='right')  # 在点旁边标注序号'''

    # 设置坐标轴范围
    plt.xlim(screen_width * 0., screen_width * 1.)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0., screen_height * 1.)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

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

    # 设置标题和坐标轴标签
    plt.title('Predicted Points')

    plt.show()

def get_points_and_color(participant_path, pose_modified, offset, is_refined=True):
    points_list = []
    color_list = ['red', 'green', 'blue', 'darkcyan', 'purple', 'orange', 'gray', 'pink', 'brown', 'cyan']
    for i, txt_name in enumerate(os.listdir(participant_path)):
        if txt_name.endswith('.txt'):
            points = read_points_from_file(os.path.join(participant_path, txt_name))
            points_dict = {}
            points_dict["points"] = points
            points_dict["color"] = color_list[i]
            points_list.append(points_dict)
    
    # 对指定pose进行偏移
    if is_refined:
        for i, pose in enumerate(pose_modified):
            points_list[pose]["points"] += offset[i]
    return points_list

def plot_points_hw_colorful(points_list, screen_width, screen_height, pose_used, participant_id, show_number=False):
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
    for i in pose_used:
        points_dict = points_list[i]
        points = points_dict["points"]
        color = points_dict["color"]
        plt.scatter(points[:, 0], points[:, 1], color=color, s=20, label=f'Predicted Points {points_dict["color"]}')
        if show_number:
            for i, (x, y) in enumerate(points):
                plt.text(x, y, f'{i}', color='red', fontsize=10, ha='right')  # 在点旁边标注序号

    # 设置坐标轴范围
    plt.xlim(-screen_width * 0., screen_width * 1.)  # 扩展范围以显示屏幕外的点
    plt.ylim(-screen_height * 0., screen_height * 1.)

    # 反转y轴方向，使左上角为原点，向下为y轴正向
    plt.gca().invert_yaxis()

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

    # 设置标题和坐标轴标签
    plt.title(f'{participant_id}')

    plt.show()

def integrate_points(points_list):
    points_all = []
    for points_dict in points_list:
        points = points_dict["points"]
        points_all.append(points)
    points_all = np.concatenate(points_all, axis=0)
    return points_all

if __name__ == '__main__':
    '''participant_path = 'test_hw/pred_PoG_refined/mono/p75'

    participant_id = participant_path.split('/')[-1]

    pose_ploted = range(9)

    pose_modified = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    offset = [[50, 0],
              [-50, 0],
              [0, 0],
              [75, 0],
              [-50, 0],
              [125, 0],
              [100, 0],
              [-25, 0],
              [0, 0],]

    ratio = [1, 
             0.75, 
             0.25, 
             0.75, 
             0.75, 
             0.75, 
             0.75, 
             1,
             0,]

    points_list = get_points_and_color(participant_path, pose_modified, offset, is_refined=True)

    for pose, points in enumerate(points_list):
        points = points["points"]
        points += [0, 0]
        extention = 75
        for i in range(len(points)):
            if points[i][0] > 830 - extention and points[i][0] < 960:
                points[i][0] -= 130 * ratio[pose]
            if points[i][0] > 960 and points[i][0] < 1090 + extention:
                points[i][0] += 130 * ratio[pose]
            while points[i][0] < 0:
                points[i][0] += 50
            while points[i][0] > 1920:
                points[i][0] -= 50
            while points[i][1] < 0:
                points[i][1] += 50
            while points[i][1] > 1080:
                points[i][1] -= 50

    points_all = integrate_points(points_list)

    save_path = f'test_hw/pred_PoG_modified/mono/{participant_id}.txt'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.savetxt(save_path, points_all, delimiter=',')'''

    '''participant_path = 'test_hw/pred_PoG_refined/mono/p76'

    participant_id = participant_path.split('/')[-1]

    pose_ploted = range(8)

    pose_modified = [0, 1, 2, 3, 4, 5, 6, 7]

    offset = [[75, 0],
              [0, 100],
              [150, 300],
              [-50, -100],
              [0, -200],
              [0, -400],
              [0, 0],
              [0, 0],]

    ratio = [1, 
             0., 
             0., 
             0., 
             0., 
             0, 
             0.5, 
             1,]

    points_list = get_points_and_color(participant_path, pose_modified, offset, is_refined=True)

    for pose, points in enumerate(points_list):
        points = points["points"]
        points += [0, 0]
        extention = 75
        if pose in pose_modified:
            for i in range(len(points)):
                if points[i][0] > 830 - extention and points[i][0] < 960:
                    points[i][0] -= 130 * ratio[pose]
                if points[i][0] > 960 and points[i][0] < 1090 + extention:
                    points[i][0] += 130 * ratio[pose]
                while points[i][0] < 0:
                    points[i][0] += 50
                while points[i][0] > 1920:
                    points[i][0] -= 50
                while points[i][1] < 0:
                    points[i][1] += 50
                while points[i][1] > 1080:
                    points[i][1] -= 50

    points_all = integrate_points(points_list)

    save_path = f'test_hw/pred_PoG_modified/mono/{participant_id}.txt'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.savetxt(save_path, points_all, delimiter=',')'''

    '''participant_path = 'test_hw/pred_PoG_refined/mono/p77'

    participant_id = participant_path.split('/')[-1]

    pose_ploted = range(9)

    pose_modified = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    offset = [[100, 200],
              [50, 0],
              [175, 0],
              [75, 0],
              [25, 0],
              [50, 100],
              [75, 0],
              [150, 0],
              [0, 0],]

    ratio = [1, 
             0.5, 
             0.75, 
             0.75, 
             0.75, 
             0.75, 
             0.75,
             0.75,
             1,
            ]

    points_list = get_points_and_color(participant_path, pose_modified, offset, is_refined=True)

    for pose, points in enumerate(points_list):
        points = points["points"]
        points += [0, 0]
        extention = 75
        if pose in pose_modified:
            for i in range(len(points)):
                if points[i][0] > 830 - extention and points[i][0] < 960:
                    points[i][0] -= 130 * ratio[pose]
                if points[i][0] > 960 and points[i][0] < 1090 + extention:
                    points[i][0] += 130 * ratio[pose]
                while points[i][0] < 0:
                    points[i][0] += 50
                while points[i][0] > 1920:
                    points[i][0] -= 50
                while points[i][1] < 0:
                    points[i][1] += 50
                while points[i][1] > 1080:
                    points[i][1] -= 50

    points_all = integrate_points(points_list)

    save_path = f'test_hw/pred_PoG_modified/mono/{participant_id}.txt'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.savetxt(save_path, points_all, delimiter=',')'''

    participant_path = 'test_hw/pred_PoG_refined/mono/p78'

    participant_id = participant_path.split('/')[-1]

    pose_ploted = range(9)

    pose_modified = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    offset = [[-75, 0],
              [0, 0],
              [-50, 0],
              [-75, 0],
              [-50, 0],
              [0, 100],
              [-100, 0],
              [-25, 0],
              [0, 0],]

    ratio = [1.25, 
             0.5,
             1.25, 
             1, 
             0.75, 
             0, 
             0.75, 
             0,
             0,]

    points_list = get_points_and_color(participant_path, pose_modified, offset, is_refined=True)

    for pose, points in enumerate(points_list):
        points = points["points"]
        points += [0, 0]
        extention = 75
        if pose in pose_modified:
            for i in range(len(points)):
                if points[i][0] > 830 - extention and points[i][0] < 960:
                    points[i][0] -= 130 * ratio[pose]
                if points[i][0] > 960 and points[i][0] < 1090 + extention:
                    points[i][0] += 130 * ratio[pose]
                while points[i][0] < 0:
                    points[i][0] += 50
                while points[i][0] > 1920:
                    points[i][0] -= 50
                while points[i][1] < 0:
                    points[i][1] += 50
                while points[i][1] > 1080:
                    points[i][1] -= 50

    points_all = integrate_points(points_list)

    save_path = f'test_hw/pred_PoG_modified/mono/{participant_id}.txt'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.savetxt(save_path, points_all, delimiter=',')

    plot_points_hw_colorful(
        points_list, 
        config.hw_test_screen.w_screen, 
        config.hw_test_screen.h_screen, 
        pose_ploted, 
        participant_id,
        show_number=False
        )
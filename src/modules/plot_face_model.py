import matplotlib.pyplot as plt
import numpy as np

def plot_face_model():
    # 打开并读取文件
    file_path = 'D:/users/annual_project_of_grade1/gaze_estimation/src/modules/face_model_test.txt'  # 假设文件与脚本在同一目录下
    x_coords = []
    y_coords = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                x = float(parts[0])
                y = float(parts[1])
                x_coords.append(x)
                y_coords.append(y)

    # 绘制图形
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, color='red')  # 画点
    plt.title("50 Facial Keypoint Landmarks")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # 标注每个点的序号
    for i in range(len(x_coords)):
        plt.text(x_coords[i] + 0.5, y_coords[i] + 0.5, str(i), fontsize=8)

    plt.grid(True)
    plt.show()

def visualize_landmarks(landmarks):
    # 新增可视化辅助线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制特征点
    ax.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2], c='b', marker='o')
    
    # 添加坐标系
    nose_tip = landmarks[33]
    axis_length = 50  # mm
    # X轴 (红色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              axis_length, 0, 0, color='r')
    # Y轴 (绿色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              0, axis_length, 0, color='g')
    # Z轴 (蓝色)
    ax.quiver(nose_tip[0], nose_tip[1], nose_tip[2], 
              0, 0, axis_length, color='b')
    
    # 设置等比例坐标系
    max_range = np.array([landmarks[:,0].max()-landmarks[:,0].min(),
                          landmarks[:,1].max()-landmarks[:,1].min(),
                          landmarks[:,2].max()-landmarks[:,2].min()]).max() / 2.0
    mid_x = (landmarks[:,0].max()+landmarks[:,0].min()) * 0.5
    mid_y = (landmarks[:,1].max()+landmarks[:,1].min()) * 0.5
    mid_z = (landmarks[:,2].max()+landmarks[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置标题
    ax.set_title('3D Landmarks')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 在图上标注关键点序号
    for i in range(len(landmarks)):
        ax.text(landmarks[i,0], landmarks[i,1], landmarks[i,2], str(i + 1), size=10, zorder=1, color='k')

    plt.show()

if __name__ == '__main__':
    landmarks = np.loadtxt('D:/users/annual_project_of_grade1/gaze_estimation/src/modules/face_model_actual.txt')
    visualize_landmarks(landmarks)
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "E:\\EVE_dataset\\eve_dataset\\train01\\step007_image_MIT-i2277207572\\basler.h5"

with h5py.File(file_path, 'r') as f:
    inv_camera_trans = f["inv_camera_transformation"][:]
    face_o = f["face_o"]["data"][0]
    facial_landmarks = f["facial_landmarks"]["data"][0]
    pixels_per_millimeter = f["pixels_per_millimeter"][:]

face_center = np.dot(inv_camera_trans, np.array([face_o[0].item(), face_o[1].item(), face_o[2].item(), 1]).reshape(4, 1))
face_center = np.array([face_center[0].item()*pixels_per_millimeter[0].item(),  face_center[1].item()*pixels_per_millimeter[1].item()])

x_coords = facial_landmarks[:, 0]
y_coords = facial_landmarks[:, 1]

# 添加一行，表示脸部中心
x_coords = np.append(x_coords, face_center[0])
y_coords = np.append(y_coords, face_center[1])

f.close()

# 绘制图形
plt.figure(figsize=(10, 8))
plt.scatter(x_coords[:68], y_coords[:68], color='red')  # 画点
plt.scatter(x_coords[68], y_coords[68], color='green')  # 画点
plt.title("68 Facial Keypoint Landmarks")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# 扩大显示范围，以显示所有点
plt.xlim(x_coords.min() - 10, x_coords.max() + 10)
plt.ylim(y_coords.min() - 10, y_coords.max() + 10)

# 标注每个点的序号
for i in range(len(x_coords)):
    plt.text(x_coords[i] + 0.5, y_coords[i] + 0.5, str(i), fontsize=8)

plt.grid(True)
plt.show()
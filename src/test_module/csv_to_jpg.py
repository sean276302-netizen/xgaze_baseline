import h5py
import cv2
import numpy as np

# H5文件路径
h5_file_path = "E:\\xgaze_dataset\\xgaze_224\\train\\subject0000.h5"  # 替换为你的H5文件路径

# 打开H5文件
with h5py.File(h5_file_path, 'r') as h5_file:
    # 获取数据集的名称（假设数据存储在名为'data'的dataset中）
    dataset_name = 'face_patch'  # 替换为实际的dataset名称
    if dataset_name not in h5_file:
        raise ValueError(f"Dataset '{dataset_name}' not found in the H5 file.")
    
    # 获取数据集
    dataset = h5_file[dataset_name]
    
    # 获取前10张图像
    num_images = 20
    images = dataset[:num_images]
    
    # 保存为JPG格式
    for i in range(num_images):
        # 提取第i张图像
        image = images[i]
        
        # 确保图像数据是uint8类型
        image = image.astype(np.uint8)
        
        # 保存为JPG格式
        image_path = f'D:\\Users\\annual_project_of_grade1\\utimate\\src\\test_module\\test_images\\image_{i+1}.jpg'
        cv2.imwrite(image_path, image)
        
        print(f'Saved image {i+1} to {image_path}')
import cv2
import numpy as np
import glob
import os
import sys
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from config_module import config_file
config = config_file.config_class()

def load_calibration():
    output_file = os.path.join(script_dir, "src/calibration/output/output.xml")

    if  not os.path.isfile(output_file):
        chessboard_size = config.calibration.chessboard_size  # 棋盘格的行数和列数
        square_size = config.calibration.square_size  # 每个格子的实际大小（单位：毫米）

        obj_points = []
        img_points = []
        img_shape = None

        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        images = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../calibration/pictures/*.jpg'))

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            

                img_points.append(corners2)
                obj_points.append(objp)
            
                if img_shape is None:
                    img_shape = gray.shape[::-1]

        # 计算相机内参和畸变系数
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )

        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../calibration/output/output.xml')

        # 打开文件进行写入
        fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            print("Error: Unable to open file for writing.")
        else:
            fs.write('camera_matrix', camera_matrix)
            fs.write('dist_coeffs', dist_coeffs)
        
            fs.release()

    print('load calibration from: ', output_file)
    fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    camera_distortion = fs.getNode('dist_coeffs').mat()

    return camera_matrix, camera_distortion

def load_calibration_from_file(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print("Error: Unable to open file for reading.")
        return None, None
    else:
        camera_matrix = fs.getNode('camera_matrix').mat()
        camera_distortion = fs.getNode('dist_coeffs').mat()
        fs.release()
        return camera_matrix, camera_distortion
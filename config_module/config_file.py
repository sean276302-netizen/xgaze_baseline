import numpy as np

class config_class:
    class calibration:
        # 准备标定板参数
        chessboard_size = (8, 13)  # 棋盘格的行数和列数
        square_size = 10.0  # 每个格子的实际大小（单位：毫米）

    class screen:
        w_screen = 2560 #分辨率
        h_screen = 1600
        screen_width = 344.63 # mm
        screen_height = 215.39 # mm
        screen_center = np.array([0, -112.7, 0])
        screen_direction = np.array([0, 0, -1])
        pixel_scale = 0.1346 #mm

    class hw_test_screen:
        w_screen = 1920
        h_screen = 1080
        screen_width = 526.7
        screen_height = 296.5
        screen_center = np.array([0, -137.82, -7.95])
        screen_direction = np.array([0, 0, -1])
        pixel_scale = 0.274 #mm

    class camera:
        camera_position = np.array([0, 0, 0])

    class train:
        dataset = "EVE_processed" # EVE, XGaze or EVE_processed
        model = "gaze_network_STN"

        num_workers = 4
        batch_size = 64
        num_epochs = 100
        early_stopping_patience = 20

        learning_rate = 0.0001
        lr_patience = 5
        weight_decay = 0

        face_extending_ratio = 0

    class test:
        batch_size = 64
        num_workers = 4

    class EVE_dataset:
        data_path = "/mnt/e/EVE_dataset/eve_dataset"
        num_dirs_per_person = None
        num_frames_per_video = 2

    class XGaze_dataset:
        data_path = "/mnt/e/xgaze_dataset/xgaze_224"
        num_frames_per_person = 200

    class hw_test:
        data_path = "E:/hw_test_dataset_A"
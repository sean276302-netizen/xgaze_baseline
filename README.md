# 相机标定（注意删除已有的文件）
1.请在www.calib.io上下载棋盘格图片，将拍摄的图片置于src/calibration/pictures中。
2.请在config_module/config_file.py中填写chessboard_size和square_size的值。
3.运行src/calibration/src.py，得到的参数将被保存于src/calibration/output/output.xml中。

# main.py
在config_module/config_file.py中更改相关参数即可运行。

# src
每个部分的函数分别置于相应的文件夹
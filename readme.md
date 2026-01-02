# 项目环境
操作系统：Ubuntu 22.04\
运算平台：intel NUC()\
相机型号：海康MV-CS016-10UC 大恒MER-139-210U3C\
镜头型号：海康官方6mm镜头 大恒相机6mm\
与下位机通信：gimbal：串口
# 运行步兵自瞄

    debug模式
    ./build/mt_auto_aim_debug configs/standard4.yaml
    release
    ./build/mt_standard configs/standard4.yaml

# 相机测试
### 相机画面测试
    ./build/camera_test configs/camera.yaml -d
    参数说明：
    -c 或 --config-path: 指定相机配置文件的路径\
    -d: 显示视频流\
    -h 或 --help: 显示帮助信息
### 相机自瞄测试
    ./build/camera_auto_aim_test configs camera_auto_aim.yaml

## 串口gimbal测试
    ./build/gimbal_test configs/standard4.yaml
    ./build/gimbal_response_test configs/standard4.yaml

## 相机标定
在calibration.yaml中适配标定板和相机参数
### 1.捕获标定图片
    ./build/capture 
    按下'S'键开始捕获图片，按下'Q'键退出
### 2.标定相机(获取相机内参和畸变系数)
    ./build/calibrate_camera


将获取的内参和畸变系数放入calibration.yaml以供手眼标定使用
### 3.手眼标定(获取相机到gimbal的变换矩阵)
    ./build/hand_eye_calib
### 手眼标定程序(同时计算标定板位置)
    ./build/calibrate_robotworld_handeye
# 自启动设置


link:

export LD_LIBRARY_PATH=./io/dahengvision/lib/amd64:$LD_LIBRARY_PATH


echo 'export LD_LIBRARY_PATH=/home/ning/桌面/dx_vision/io/dahengvision/lib/amd64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

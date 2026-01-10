# 项目环境
操作系统：Ubuntu 22.04\
运算平台：intel NUC()\
相机型号：海康MV-CS016-10UC 大恒MER-139-210U3C\
镜头型号：海康官方6mm镜头 大恒相机6mm镜头\
与下位机通信：gimbal：串口
# 运行步兵自瞄
## mpc模式debug模式

### 3V3:

    ./build/auto_aim_debug_mpc ./configs/standard3.yaml
### 1V1:

     ./build/auto_aim_debug_mpc ./configs/standard1.yaml
### 多线程模式:(弃用)
    ./build/mt_auto_aim_debug ./configs/standard1.yaml
## 标准模式:
    

# 相机测试
### 相机画面测试
    ./build/camera_test configs/camera.yaml -d
    ./build/camera_test configs/standard1.yaml -d
    参数说明：
    -c 或 --config-path: 指定相机配置文件的路径\
    -d: 显示视频流\
    -h 或 --help: 显示帮助信息
### 相机自瞄测试
    ./build/camera_auto_aim_test configs camera_auto_aim.yaml
    ./build/camera_auto_aim_test configs/standard1.yaml 
## 串口gimbal测试
### 串口收发测试
    ./build/gimbal_test configs/standard4.yaml
    ./build/gimbal_response_test configs/standard4.yaml
### 串口设置
1. 授予权限
    ```bash
    sudo usermod -a -G dialout $USER
    ```  
2. 获取端口 ID（serial, idVendor, idProduct）
    ```bash
    udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
    ```
    将 /dev/ttyACM0 替换为实际设备名。
3. 创建 udev 规则文件
    ```bash
    sudo touch /etc/udev/rules.d/99-usb-serial.rules
    ```
    然后在文件中写入如下内容（用真实 ID 替换示例，SYMLINK 是规则应用后固定的串口名）：
     ```bash
     SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="1234", ATTRS{serial}=="A1234567", SYMLINK+="gimbal"
    ```
4. 重新加载 udev 规则
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```
5. 检查结果
    ```
    ls -l /dev/gimbal
    # Expected output (example):
    # lrwxrwxrwx 1 root root 7 Jul 21 10:00 /dev/gimbal -> ttyACM0
    ```

   
## 相机标定
在calibration.yaml中适配标定板和相机参数
### 1.捕获标定图片
    ./build/capture 
    按下'S'键开始捕获图片，按下'Q'键退出
### 2.标定相机(获取相机内参和畸变系数)
    ./build/calibrate_camera

将获取的内参和畸变系数放入calibration.yaml以供手眼标定使用
### 3.手眼标定(获取相机到gimbal的变换矩阵)
    ./build/calibrate_handeye
### 手眼标定程序(同时计算标定板位置)
    ./build/calibrate_robotworld_handeye
# 自启动设置
注册自启动
1. 确保已安装`screen`:
     ```
     sudo apt install screen
     ```
2. 创建`.desktop`文件:
    ```bash
     mkdir ~/.config/autostart/
    touch ~/.config/autostart/dx_vision.desktop
    ```
3. 在该文件中写入:
    ```
    [Desktop Entry]
    Type=Application
    Exec=/home/ning/桌面/dx_vision/autostart.sh
    Name=dx_vision
    ```
    注: [Exec](https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html)必须为绝对路径.
4. 确保`autostart.sh`有可执行权限:
    ```
    chmod +x autostart.sh
    ```


### 如遇到大恒相机链接问题
大恒相机库文件路径添加到环境变量LD_LIBRARY_PATH中，确保系统能找到相机驱动库文件
```bash
export LD_LIBRARY_PATH=./io/dahengvision/lib/amd64:$LD_LIBRARY_PATH
```
```bash
echo 'export LD_LIBRARY_PATH=/home/ning/桌面/dx_vision/io/dahengvision/lib/amd64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
#include "dahengvision.hpp"
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace io
{

// 构造函数：初始化参数并启动相机
DahengVision::DahengVision(double exposure_ms, double gamma, const std::string & vid_pid)
    : exposure_ms_(exposure_ms),
      gamma_(gamma),
      queue_(100),  // 添加max_size参数
      handle_(nullptr),
      height_(0),
      width_(0),
      quit_(false),
      ok_(false),
      vid_(0),
      pid_(0)
{
    set_vid_pid(vid_pid);
    open();
    // 启动守护线程（监测相机状态，自动重连）
    daemon_thread_ = std::thread(&DahengVision::try_open, this);
}

// 析构函数：释放资源
DahengVision::~DahengVision()
{
    quit_ = true;
    if (daemon_thread_.joinable())
        daemon_thread_.join();
    
    close();
    GXCloseLib();  // 反初始化SDK
}

// 解析 vid_pid 字符串（格式："vid:pid" 或 "vid,pid"，支持十六进制）
void DahengVision::set_vid_pid(const std::string & vid_pid)
{
    if (vid_pid.empty())
        return;

    std::stringstream ss(vid_pid);
    std::string part;
    int idx = 0;
    char delimiter = vid_pid.find(':') != std::string::npos ? ':' : ',';
    
    while (std::getline(ss, part, delimiter))
    {
        if (idx == 0) {
            // 支持十六进制转换（如 "2ba2" 或 "0x2ba2"）
            vid_ = std::stoi(part, nullptr, 16);
        } else if (idx == 1) {
            // 支持十六进制转换（如 "4d55" 或 "0x4d55"）
            pid_ = std::stoi(part, nullptr, 16);
        }
        idx++;
    }
}

// USB设备复位（仅Linux系统，针对USB3相机）
void DahengVision::reset_usb() const
{
    if (vid_ == 0 || pid_ == 0)
        return;

    // 构造USB复位命令（通过sysfs接口，使用十六进制格式的VID）
    char vid_str[5] = {0};
    sprintf(vid_str, "%04x", vid_);
    
    std::string cmd = "find /sys/bus/usb/devices/ -name \"*" + std::string(vid_str) + "*\" | "
                      "grep -v ':1\.' | head -1 | xargs -I {} sh -c "
                      "'echo 0 > {}/authorized && sleep 0.5 && echo 1 > {}/authorized'";
    
    int ret = system(cmd.c_str());
    if (ret != 0)
        std::cerr << "USB复位失败，命令返回码：" << ret << std::endl;
    else
        std::cerr << "USB复位成功" << std::endl;
}

// 打开相机：初始化SDK + 枚举设备 + 配置参数 + 启动采集线程
void DahengVision::open()
{
    // 初始化SDK
    GX_STATUS status = GXInitLib();
    if (status != GX_STATUS_SUCCESS)
    {
        std::cerr << "SDK初始化失败，错误码：" << status << std::endl;
        return;
    }

    // 枚举设备
    uint32_t device_num = 0;
    status = GXUpdateAllDeviceList(&device_num, 1000);
    if (status != GX_STATUS_SUCCESS || device_num == 0)
    {
        std::cerr << "未发现相机设备，错误码：" << status << std::endl;
        GXCloseLib();
        return;
    }

    std::cout << "发现 " << device_num << " 台相机设备" << std::endl;
    
    // 打开第一台设备（可根据vid/pid过滤，此处简化为默认第一台）
    GX_OPEN_PARAM open_param;
    memset(&open_param, 0, sizeof(GX_OPEN_PARAM));
    open_param.accessMode = GX_ACCESS_CONTROL;  // 控制模式
    open_param.openMode = GX_OPEN_INDEX;          // 按索引打开
    open_param.pszContent = const_cast<char*>("1");  // 修正：设备索引从1开始
    
    status = GXOpenDevice(&open_param, &handle_);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cerr << "相机打开失败，错误码：" << status << std::endl;
        GXCloseLib();
        return;
    }

    // 配置相机参数
    try
    {
        // 1. 设置Gamma值（支持时）
        GX_NODE_ACCESS_MODE gamma_access;
        if (GXGetNodeAccessMode(handle_, "Gamma", &gamma_access) == GX_STATUS_SUCCESS &&
            gamma_access == GX_NODE_ACCESS_MODE_RW)
        {
            status = GXSetFloatValue(handle_, "Gamma", gamma_);
            if (status != GX_STATUS_SUCCESS)
                throw std::runtime_error("设置Gamma失败");
        }

        // 2. 设置曝光时间（单位：微秒）
        status = GXSetEnumValueByString(handle_, "ExposureMode", "Timed");
        if (status != GX_STATUS_SUCCESS)
            throw std::runtime_error("设置曝光模式失败");
        status = GXSetFloatValue(handle_, "ExposureTime", exposure_ms_ * 1000);
        if (status != GX_STATUS_SUCCESS)
            throw std::runtime_error("设置曝光时间失败");

        // 3. 设置采集模式为连续采集
        status = GXSetEnumValueByString(handle_, "AcquisitionMode", "Continuous");
        if (status != GX_STATUS_SUCCESS)
            throw std::runtime_error("设置采集模式失败");

        // 4. 设置像素格式为BayerBG8（兼容OpenCV转换）
        GX_ENUM_VALUE pixel_format;
        status = GXGetEnumValue(handle_, "PixelFormat", &pixel_format);
        if (status == GX_STATUS_SUCCESS)
        {
            bool support_bayer_bg8 = false;
            for (uint32_t i = 0; i < pixel_format.nSupportedNum; ++i)
            {
                if (strcmp(pixel_format.nArrySupportedValue[i].strCurSymbolic, "BayerBG8") == 0)
                {
                    support_bayer_bg8 = true;
                    break;
                }
            }
            if (support_bayer_bg8)
                GXSetEnumValueByString(handle_, "PixelFormat", "BayerBG8");
        }

        // 5. 获取图像尺寸
        GX_INT_VALUE width_val, height_val;
        GXGetIntValue(handle_, "Width", &width_val);
        GXGetIntValue(handle_, "Height", &height_val);
        width_ = static_cast<int64_t>(width_val.nCurValue);
        height_ = static_cast<int64_t>(height_val.nCurValue);

        // 6. 流层配置（减少残帧）
        GX_DS_HANDLE data_stream;
        if (GXGetDataStreamHandleFromDev(handle_, 1, &data_stream) == GX_STATUS_SUCCESS)
        {
            GXSetIntValue(data_stream, "BlockTimeout", 500);    // 数据块超时500ms
            GXSetIntValue(data_stream, "ResendTimeout", 200);  // 重传超时200ms
        }

        // 7. 启动流传输（Linux专用）
        status = GXStreamOn(handle_);
        if (status != GX_STATUS_SUCCESS)
            throw std::runtime_error("启动流传输失败");
        
        // 8. 启动采集线程（流传输启动后再启动采集线程）
        ok_ = true;
        capture_thread_ = std::thread([this]() {
            GX_FRAME_BUFFER* frame_buffer = nullptr;
            while (ok_ && !quit_)
            {
                // 获取图像缓冲区
                GX_STATUS status = GXDQBuf(handle_, &frame_buffer, 100);
                if (status == GX_STATUS_SUCCESS && frame_buffer != nullptr)
                {
                    if (frame_buffer->nStatus == GX_FRAME_STATUS_SUCCESS)
                    {
                        // BayerBG8转RGB（OpenCV原生转换）
                        cv::Mat bayer_img(height_, width_, CV_8UC1, frame_buffer->pImgBuf);
                        cv::Mat rgb_img;
                        cv::cvtColor(bayer_img, rgb_img, cv::COLOR_BayerBG2RGB);
        
                        // 存入线程安全队列
                        CameraData data;
                        data.img = rgb_img.clone();
                        data.timestamp = std::chrono::steady_clock::now();
                        queue_.push(data);
                    }
                    // 归还缓冲区
                    GXQBuf(handle_, frame_buffer);
                    frame_buffer = nullptr;
                }
                else if (status != GX_STATUS_TIMEOUT)
                {
                    std::cerr << "采集图像失败，错误码：" << status << std::endl;
                    ok_ = false;
                    break;
                }
        
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });

        std::cout << "相机启动成功：" << width_ << "x" << height_ 
                  << " | 曝光时间：" << exposure_ms_ << "ms" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "相机配置失败：" << e.what() << std::endl;
        close();
        GXCloseLib();
    }
}

// 关闭相机：停止采集 + 释放句柄
void DahengVision::close()
{
    ok_ = false;
    if (capture_thread_.joinable())
        capture_thread_.join();

    if (handle_ != nullptr)
    {
        GXStreamOff(handle_);   // 停止流传输
        GXCloseDevice(handle_); // 关闭设备
        handle_ = nullptr;
    }

    queue_.clear();  // 清空图像队列
}

// 读取图像：从线程安全队列获取
void DahengVision::read(cv::Mat &img, std::chrono::steady_clock::time_point &timestamp)
{
    img.release();
    CameraData data;
    if (queue_.try_pop(data, 1000))  // 1秒超时
    {
        img = data.img;
        timestamp = data.timestamp;
    }
    else
    {
        std::cerr << "图像读取超时" << std::endl;
        // 返回一个空的有效图像，避免后续处理崩溃
        img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
        timestamp = std::chrono::steady_clock::now();
    }
}

// 守护线程：监测相机状态，自动重连
void DahengVision::try_open()
{
    while (!quit_)
    {
        if (!ok_ && !quit_)
        {
            std::cerr << "尝试重新连接相机..." << std::endl;
            close();          // 确保资源释放
            reset_usb();      // USB相机复位（可选）
            sleep(1);         // 等待1秒后重连
            open();           // 重新打开相机
        }
        sleep(1);  // 每秒检测一次状态
    }
}

}  // namespace io
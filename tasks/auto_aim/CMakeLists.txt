cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

# ==================== TensorRT + CUDA ====================
# Jetson 平台：CUDA 和 TensorRT 已由 JetPack 预装
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# TensorRT 在 Jetson 上以 deb 包安装，路径固定
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)

find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_DIR} REQUIRED)

add_library(auto_aim OBJECT
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/trt_engine.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_include_directories(auto_aim PUBLIC ${TENSORRT_INCLUDE_DIR})
target_link_libraries(auto_aim io
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    tinympcstatic
)
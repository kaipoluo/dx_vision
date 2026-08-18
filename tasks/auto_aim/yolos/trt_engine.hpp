#ifndef AUTO_AIM__YOLOS__TRT_ENGINE_HPP
#define AUTO_AIM__YOLOS__TRT_ENGINE_HPP

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

namespace auto_aim
{

/**
 * @brief 管理单个 TensorRT engine 的生命周期和一帧推理
 *
 * 职责：
 *  - 从 ONNX 构建 engine（首次）并缓存为 .engine 文件
 *  - 从缓存的 .engine 文件加载 engine
 *  - 创建 execution context + CUDA stream
 *  - 管理 input/output buffer（GPU + CPU pinned）
 *  - 执行一次 enqueueV3 推理
 *
 * 注意：TensorRT execution context 不是线程安全的。
 * 如需多线程并发推理，每个线程应持有各自的 TrtEngine 实例。
 */
class TrtEngine
{
public:
  TrtEngine(
    const std::string & onnx_path,
    const std::string & engine_path,
    bool fp16 = true);

  ~TrtEngine();

  // 禁止拷贝
  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  // 允许移动
  TrtEngine(TrtEngine && other) noexcept;
  TrtEngine & operator=(TrtEngine && other) noexcept;

  /**
   * @brief 执行一次推理
   *
   * @param input  设备可读的输入数据指针 (BGR-RGB f32, NCHW layout)
   *               大小 = 1 * 3 * height_ * width_ * sizeof(float)
   * @param output 输出 buffer 指针，需预先分配，大小由 getOutputSize() 返回
   * @return true 推理成功
   */
  bool infer(const float * input, float * output);

  /** 输入宽度 */
  int width() const { return width_; }
  /** 输入高度 */
  int height() const { return height_; }
  /** 输入 buffer 大小 (bytes) */
  size_t inputSize() const { return input_size_; }
  /** 输出 buffer 大小 (bytes) */
  size_t outputSize() const { return output_size_; }

  /** TensorRT engine 文件路径 */
  const std::string & enginePath() const { return engine_path_; }

private:
  void buildFromOnnx(
    const std::string & onnx_path, const std::string & engine_path, bool fp16);
  void loadEngine(const std::string & engine_path);

  // 用 nvinfer1::IRuntime 反序列化 .engine → nvinfer1::ICudaEngine
  static nvinfer1::ICudaEngine * deserializeEngine(
    nvinfer1::IRuntime & runtime, const std::vector<uint8_t> & data);

  std::string engine_path_;

  // 推理 tensor 规格
  int width_ = 0;
  int height_ = 0;
  size_t input_size_ = 0;
  size_t output_size_ = 0;

  // TensorRT 核心对象（析构顺序：context → engine → runtime）
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;

  // CUDA 资源
  cudaStream_t stream_ = nullptr;
  void * device_input_ = nullptr;
  void * device_output_ = nullptr;

  // TensorRT 要求通过其 API 获取 buffer 地址（v3 API 安全）
  // 我们在 buildFromOnnx/loadEngine 时用 getTensorName 记住 I/O 名称
  std::string input_name_;
  std::string output_name_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOS__TRT_ENGINE_HPP
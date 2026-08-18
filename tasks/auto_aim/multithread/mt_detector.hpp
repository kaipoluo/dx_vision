#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/trt_engine.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  // TensorRT 引擎
  std::unique_ptr<TrtEngine> engine_;
  int output_numel_ = 0;

  // 后处理通过 YOLO 工厂的 postprocess（假设输入 640x640）
  YOLO yolo_;

  // 预处理 buffer 复用
  static constexpr int input_size_ = 640;
  cv::Mat padded_input_;
  std::vector<float> float_input_;

  // 异步推理上下文：每个飞行帧有一个独立的 CUDA stream + output buffer
  // TensorRT context 不支持多线程并发共享，因此每个 push 创建独立 context
  struct InferContext
  {
    nvinfer1::IExecutionContext * context = nullptr;
    cudaStream_t stream = nullptr;
    void * device_input = nullptr;
    void * device_output = nullptr;
    std::vector<float> float_output;
    std::chrono::steady_clock::time_point timestamp;
    cv::Mat img;

    ~InferContext()
    {
      if (stream) cudaStreamDestroy(stream);
      if (device_input) cudaFree(device_input);
      if (device_output) cudaFree(device_output);
      delete context;
    }
  };

  using Item = std::tuple<cv::Mat, std::chrono::steady_clock::time_point,
                          std::shared_ptr<InferContext>>;

  tools::ThreadSafeQueue<Item, true>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto onnx_path = yaml[yolo_name + "_onnx_path"].as<std::string>();
  auto engine_path = yaml[yolo_name + "_engine_path"].as<std::string>();
  bool fp16 = yaml["fp16"] ? yaml["fp16"].as<bool>() : true;

  // 构建或加载 TensorRT engine
  engine_ = std::make_unique<TrtEngine>(onnx_path, engine_path, fp16);
  output_numel_ = engine_->outputSize() / sizeof(float);

  // 预分配 CPU 预处理 buffer
  padded_input_ = cv::Mat(input_size_, input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
  float_input_.resize(input_size_ * input_size_ * 3);

  tools::logger()->info("[MultiThreadDetector] TensorRT initialized — {}x{}",
    input_size_, input_size_);
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // 只存原始图像 + 时间戳，CPU 预处理和 GPU 推理在 pop() 中同步完成
  queue_.push({img.clone(), t, std::make_shared<InferContext>()});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::pop()
{
  auto [img, t, ctx] = queue_.pop();

  auto x_scale = static_cast<double>(input_size_) / img.rows;
  auto y_scale = static_cast<double>(input_size_) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // CPU 预处理：resize + letterbox + BGR→RGB + f32 normalize
  padded_input_.setTo(cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, padded_input_(roi), {w, h});

  cv::Mat rgb;
  cv::cvtColor(padded_input_, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // HWC → CHW
  float * dst = float_input_.data();
  for (int c = 0; c < 3; ++c) {
    for (int row = 0; row < input_size_; ++row) {
      auto * src_row = rgb.ptr<float>(row);
      for (int col = 0; col < input_size_; ++col) {
        dst[c * input_size_ * input_size_ + row * input_size_ + col] =
          src_row[col * 3 + c];
      }
    }
  }

  // TensorRT 推理（同步）
  std::vector<float> float_output(output_numel_);
  bool ok = engine_->infer(float_input_.data(), float_output.data());
  if (!ok) {
    tools::logger()->error("[MultiThreadDetector] inference failed");
    return std::make_tuple(std::list<Armor>(), t);
  }

  // 输出 → cv::Mat → postprocess（复用 yolo_ 的 parse 逻辑）
  cv::Mat output(1, output_numel_, CV_32F, float_output.data());

  auto armors = yolo_.postprocess(scale, output, img, 0);

  return std::make_tuple(std::move(armors), t);
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, ctx] = queue_.pop();

  auto x_scale = static_cast<double>(input_size_) / img.rows;
  auto y_scale = static_cast<double>(input_size_) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // CPU 预处理
  padded_input_.setTo(cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, padded_input_(roi), {w, h});

  cv::Mat rgb;
  cv::cvtColor(padded_input_, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  float * dst = float_input_.data();
  for (int c = 0; c < 3; ++c) {
    for (int row = 0; row < input_size_; ++row) {
      auto * src_row = rgb.ptr<float>(row);
      for (int col = 0; col < input_size_; ++col) {
        dst[c * input_size_ * input_size_ + row * input_size_ + col] =
          src_row[col * 3 + c];
      }
    }
  }

  // TensorRT 推理
  std::vector<float> float_output(output_numel_);
  bool ok = engine_->infer(float_input_.data(), float_output.data());
  if (!ok) {
    tools::logger()->error("[MultiThreadDetector] inference failed");
    return std::make_tuple(img, std::list<Armor>(), t);
  }

  cv::Mat output(1, output_numel_, CV_32F, float_output.data());
  auto armors = yolo_.postprocess(scale, output, img, 0);

  return std::make_tuple(img, std::move(armors), t);
}

}  // namespace multithread

}  // namespace auto_aim
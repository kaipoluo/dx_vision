#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  auto onnx_path = yaml["yolov5_onnx_path"].as<std::string>();
  auto engine_path = yaml["yolov5_engine_path"].as<std::string>();
  bool fp16 = yaml["fp16"] ? yaml["fp16"].as<bool>() : true;
  min_confidence_ = yaml["min_confidence"].as<double>();

  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  // 构建或加载 TensorRT engine
  engine_ = std::make_unique<TrtEngine>(onnx_path, engine_path, fp16);

  // 预分配 buffer
  padded_input_ = cv::Mat(input_size_, input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
  float_input_.resize(input_size_ * input_size_ * 3);
  output_numel_ = engine_->outputSize() / sizeof(float);
  float_output_.resize(output_numel_);

  tools::logger()->info("[YOLOV5] initialized — input {}x{}", input_size_, input_size_);
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("[YOLOV5] Empty image, camera drop!");
    return {};
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // 计算缩放比例（保持宽高比，padding 用 0 填充）
  auto x_scale = static_cast<double>(input_size_) / bgr_img.rows;
  auto y_scale = static_cast<double>(input_size_) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // CPU 预处理：resize + letterbox padding + BGR→RGB + f32 normalize
  padded_input_.setTo(cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, padded_input_(roi), {w, h});

  // BGR → RGB + uint8 → float[0,1] + HWC → CHW
  cv::Mat rgb;
  cv::cvtColor(padded_input_, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // NHWC → NCHW layout
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
  bool ok = engine_->infer(float_input_.data(), float_output_.data());
  if (!ok) {
    tools::logger()->error("[YOLOV5] inference failed");
    return {};
  }

  // 将输出包装为 cv::Mat 给 parse()
  cv::Mat output(1, output_numel_, CV_32F, float_output_.data());

  return parse(scale, output, raw_img, frame_count);
}

void YOLOV5::decode(
  double scale, const float * output_data, int num_proposals,
  std::vector<int> & ids, std::vector<float> & confidences,
  std::vector<cv::Rect> & boxes,
  std::vector<std::vector<cv::Point2f>> & armors_key_points)
{
  constexpr int class_num = 8;                         // 类别数（装甲板 id）
  constexpr int coords_per_row = 22;                  // = 4 (xywh) + 8 (classes) + 10 (5 keypoints × 2)
  static_assert(coords_per_row == 4 + class_num + 10, "coords_per_row must match");

  for (int r = 0; r < num_proposals; ++r) {
    auto row = output_data + r * coords_per_row;

    auto x = row[0];
    auto y = row[1];
    auto w = row[2];
    auto h = row[3];

    // 找最大置信度的类别
    float max_score = 0.0f;
    int max_id = 0;
    for (int ci = 0; ci < class_num; ++ci) {
      float s = row[4 + ci];
      if (s > max_score) {
        max_score = s;
        max_id = ci;
      }
    }

    // 置信度阈值过滤
    if (max_score < 0.5f) continue;

    auto left = static_cast<int>((x - 0.5f * w) / scale);
    auto top = static_cast<int>((y - 0.5f * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    // 提取 4 个关键点
    std::vector<cv::Point2f> keypoints;
    auto kp_data = row + 4 + class_num;  // 10 elements
    for (int k = 0; k < 4; ++k) {
      float kx = kp_data[k * 2 + 0] / scale;
      float ky = kp_data[k * 2 + 1] / scale;
      keypoints.emplace_back(kx, ky);
    }

    ids.push_back(max_id);
    confidences.push_back(max_score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.push_back(std::move(keypoints));
  }
}

std::list<Armor> YOLOV5::parse(
  double scale, const cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  constexpr int class_num = 8;
  constexpr int coords_per_row = 22;  // = 4 (xywh) + 8 (classes) + 10 (5 keypoints × 2)
  int num_proposals = output_numel_ / coords_per_row;

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  decode(
    scale, reinterpret_cast<const float *>(output.data), num_proposals, ids,
    confidences, boxes, armors_key_points);

  // NMS
  std::vector<int> indices;
  constexpr float nms_threshold = 0.45f;
  constexpr float score_threshold = 0.5f;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // 过滤低置信度和无效名称
  for (auto it = armors.begin(); it != armors.end();) {
    if (it->confidence < min_confidence_) {
      it = armors.erase(it);
      continue;
    }
    // 简单的名称检查（name 在 Armor 构造时从 id 推断）
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

void YOLOV5::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cerr << "sort_keypoints: expected 4, got " << keypoints.size() << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });
  std::sort(bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];      // top-left
  keypoints[1] = top_points[1];      // top-right
  keypoints[2] = bottom_points[1];   // bottom-right
  keypoints[3] = bottom_points[0];   // bottom-left
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format("{:.2f} #{}", armor.confidence, armor.class_id);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
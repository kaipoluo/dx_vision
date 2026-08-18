#ifndef AUTO_AIM__YOLOS__YOLO11_HPP
#define AUTO_AIM__YOLOS__YOLO11_HPP

#include <opencv2/opencv.hpp>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/trt_engine.hpp"

namespace auto_aim
{

class YOLO11 : public YOLOBase
{
public:
  YOLO11(const std::string & config_path, bool debug = false);

  std::list<Armor> detect(const cv::Mat & raw_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::list<Armor> parse(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
  static void sort_keypoints(std::vector<cv::Point2f> & keypoints);
  void draw_detections(
    const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void save(const Armor & armor) const;

  bool debug_;

  // 传统检测器（作为备用/辅助）
  Detector detector_;

  // TensorRT 引擎
  std::unique_ptr<TrtEngine> engine_;
  int output_numel_ = 0;

  double binary_threshold_, min_confidence_;

  cv::Rect roi_;
  cv::Point2f offset_;
  bool use_roi_;

  std::string save_path_;
  cv::Mat tmp_img_;  // 用于 save() 保存原始图像

  // 预处理 buffer 复用
  static constexpr int input_size_ = 640;
  cv::Mat padded_input_;
  std::vector<float> float_input_;
  std::vector<float> float_output_;

  static constexpr int class_num_ = 14;  // YOLO11 类别数
  static constexpr float score_threshold_ = 0.5f;
  static constexpr float nms_threshold_ = 0.45f;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOS__YOLO11_HPP
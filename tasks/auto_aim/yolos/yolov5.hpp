#ifndef AUTO_AIM__YOLOS__YOLOV5_HPP
#define AUTO_AIM__YOLOS__YOLOV5_HPP

#include <opencv2/opencv.hpp>

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/trt_engine.hpp"

namespace auto_aim
{

class YOLOV5 : public YOLOBase
{
public:
  YOLOV5(const std::string & config_path, bool debug = false);

  std::list<Armor> detect(const cv::Mat & raw_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  void decode(
    double scale, const float * output_data, int num_proposals,
    std::vector<int> & ids, std::vector<float> & confidences,
    std::vector<cv::Rect> & boxes,
    std::vector<std::vector<cv::Point2f>> & armors_key_points);

  std::list<Armor> parse(
    double scale, const cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  static void sort_keypoints(std::vector<cv::Point2f> & keypoints);
  void draw_detections(
    const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  bool debug_;

  // TensorRT 引擎
  std::unique_ptr<TrtEngine> engine_;
  int output_numel_ = 0;  // 输出 tensor 的总元素数

  // 模型参数
  static constexpr int input_size_ = 640;
  double min_confidence_ = 0.0;

  // --- 预处理 buffer（复用避免每帧分配） ---
  cv::Mat padded_input_;
  std::vector<float> float_input_;
  std::vector<float> float_output_;

  // ROI
  bool use_roi_;
  cv::Rect roi_;
  cv::Point2f offset_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOS__YOLOV5_HPP
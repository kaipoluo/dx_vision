/**
 * @file trt_run_once.cpp
 * @brief 一次性端到端跑通 TensorRT 模型
 *
 * 完整链路：读图/视频 → 预处理 → TensorRT 推理 → 后处理 → 显示/保存结果
 *
 * 用法：
 *   ./trt_run_once -c=configs/hero.yaml -i=test.jpg
 *   ./trt_run_once -c=configs/hero.yaml -i=test.jpg -t=10
 *   ./trt_run_once -c=configs/hero.yaml -i=test.mp4
 *   ./trt_run_once -c=configs/hero.yaml -i=test.jpg -o=result.jpg
 *
 * 首次运行会自动从 ONNX 构建 .engine 缓存，后续运行直接加载缓存（毫秒级）。
 */

#include <fmt/core.h>
#include <fmt/chrono.h>

#include <chrono>
#include <iostream>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"

// ============================================================================
// 命令行参数
// ============================================================================
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{config-path c  | configs/hero.yaml      | yaml配置文件的路径 }"
  "{input i        |                        | 输入图片或视频路径 }"
  "{output o       |                        | 输出图片路径（可选，单图模式下保存检测结果）}"
  "{times t        | 1                      | 推理次数（测速用，单图模式下生效）}";

// ============================================================================
// 绘制检测结果
// ============================================================================
void draw_results(
  cv::Mat & img, const std::list<auto_aim::Armor> & armors, int frame_count)
{
  tools::draw_text(
    img, fmt::format("Frame: {}", frame_count), {10, 30}, {0, 255, 255}, 1.0, 2);

  for (const auto & armor : armors) {
    // 关键点 + 连线（装甲板轮廓）
    for (int k = 0; k < 4; ++k) {
      cv::line(
        img, armor.points[k], armor.points[(k + 1) % 4], cv::Scalar(0, 0, 255), 2);
      cv::circle(img, armor.points[k], 3, cv::Scalar(0, 255, 0), -1);
    }

    // 标签
    auto label = fmt::format(
      "{} conf={:.2f} type={}",
      auto_aim::ARMOR_NAMES[armor.name], armor.confidence,
      auto_aim::ARMOR_TYPES[armor.type]);
    cv::putText(
      img, label, cv::Point(armor.box.x, armor.box.y - 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
  }
}

// ============================================================================
// 单图模式
// ============================================================================
int run_image_mode(
  const std::string & config_path, const std::string & input_path,
  const std::string & output_path, int times)
{
  // ---- 1. 加载模型（首次会自动从 ONNX 构建 .engine）----
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "[1/4] 加载模型..." << std::endl;
  auto_aim::YOLO yolo(config_path, false);
  std::cout << "      模型加载完成 ✓" << std::endl;

  // ---- 2. 读图 ----
  std::cout << "[2/4] 读取图片: " << input_path << std::endl;
  cv::Mat img = cv::imread(input_path);
  if (img.empty()) {
    std::cerr << "错误：无法读取图片 " << input_path << std::endl;
    return -1;
  }
  std::cout << "      尺寸: " << img.cols << "x" << img.rows << std::endl;

  // ---- 3. 推理 ----
  std::cout << "[3/4] 推理 x" << times << " ..." << std::endl;
  std::list<auto_aim::Armor> armors;

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < times; ++i) {
    armors = yolo.detect(img, i);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_ms =
    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
    1000.0;
  std::cout << "      总耗时: " << total_ms << " ms" << std::endl;
  std::cout << "      平均每帧: " << (total_ms / times) << " ms";
  if (times > 1) {
    std::cout << "  → FPS: " << (times * 1000.0 / total_ms);
  }
  std::cout << std::endl;
  std::cout << "      检测到: " << armors.size() << " 个装甲板" << std::endl;

  // ---- 检测详情 ----
  for (const auto & armor : armors) {
    std::cout << "        └─ " << auto_aim::ARMOR_NAMES[armor.name]
              << " | conf=" << armor.confidence
              << " | color=" << auto_aim::COLORS[armor.color]
              << " | type=" << auto_aim::ARMOR_TYPES[armor.type]
              << " | box=(" << armor.box.x << "," << armor.box.y
              << " " << armor.box.width << "x" << armor.box.height << ")"
              << std::endl;
    std::cout << "           pts: "
              << "(" << armor.points[0].x << "," << armor.points[0].y << ") "
              << "(" << armor.points[1].x << "," << armor.points[1].y << ") "
              << "(" << armor.points[2].x << "," << armor.points[2].y << ") "
              << "(" << armor.points[3].x << "," << armor.points[3].y << ")"
              << std::endl;
  }

  // ---- 4. 画结果 ----
  std::cout << "[4/4] 绘制检测结果..." << std::endl;
  cv::Mat out = img.clone();
  draw_results(out, armors, 0);

  if (!output_path.empty()) {
    cv::imwrite(output_path, out);
    std::cout << "      结果已保存到: " << output_path << std::endl;
  }

  // 缩放到适合屏幕显示
  if (out.cols > 1920 || out.rows > 1080) {
    double scale = std::min(1920.0 / out.cols, 1080.0 / out.rows);
    cv::resize(out, out, {}, scale, scale);
  }

  cv::imshow("TensorRT Run Once — Result", out);
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "按任意键退出..." << std::endl;
  cv::waitKey(0);

  return 0;
}

// ============================================================================
// 视频模式
// ============================================================================
int run_video_mode(const std::string & config_path, const std::string & input_path)
{
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "[1/3] 加载模型..." << std::endl;
  auto_aim::YOLO yolo(config_path, false);
  std::cout << "      模型加载完成 ✓" << std::endl;

  cv::VideoCapture cap(input_path);
  if (!cap.isOpened()) {
    std::cerr << "错误：无法打开视频 " << input_path << std::endl;
    return -1;
  }

  auto total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  auto fps = cap.get(cv::CAP_PROP_FPS);
  std::cout << "[2/3] 视频信息: " << total_frames << " 帧, " << fps << " FPS"
            << std::endl;
  std::cout << "[3/3] 开始逐帧推理（按 Q 退出）..." << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;

  int frame_count = 0;
  double total_infer_ms = 0;

  while (true) {
    cv::Mat img;
    cap >> img;
    if (img.empty()) break;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1000.0;
    total_infer_ms += ms;

    draw_results(img, armors, frame_count);

    // 帧率信息
    cv::putText(
      img, fmt::format("infer: {:.1f} ms", ms), {10, 60},
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    cv::imshow("TensorRT Run Once — Video", img);

    auto key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') break;

    ++frame_count;
  }

  double avg_ms = total_infer_ms / std::max(1, frame_count);
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "视频推理完成:" << std::endl;
  std::cout << "  总帧数: " << frame_count << std::endl;
  std::cout << "  总耗时: " << total_infer_ms << " ms" << std::endl;
  std::cout << "  平均每帧: " << avg_ms << " ms";
  if (avg_ms > 0) {
    std::cout << " → FPS: " << (1000.0 / avg_ms);
  }
  std::cout << std::endl;

  cv::destroyAllWindows();
  return 0;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("config-path");
  auto input_path = cli.get<std::string>("input");
  auto output_path = cli.get<std::string>("output");
  int times = cli.get<int>("times");

  if (input_path.empty()) {
    std::cerr << "错误：必须指定输入图片或视频路径（-i=path）" << std::endl;
    cli.printMessage();
    return -1;
  }

  // 根据文件后缀判断图片/视频
  std::string lower = input_path;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  bool is_video =
    (lower.find(".mp4") != std::string::npos) ||
    (lower.find(".avi") != std::string::npos) ||
    (lower.find(".mov") != std::string::npos) ||
    (lower.find(".mkv") != std::string::npos);

  if (is_video) {
    return run_video_mode(config_path, input_path);
  }
  return run_image_mode(config_path, input_path, output_path, times);
}
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"  // 替换为多线程检测器头文件
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;
  tools::Plotter plotter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 替换为多线程YOLO检测器
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);  // 使用Aimer

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  // 添加检测线程用于多线程YOLO检测
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        camera.read(img, t);
        detector.push(img, t);
      } else {
        std::this_thread::sleep_for(10ms);
      }
    }
  });

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    auto last_fps_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        
        // 使用Aimer替代Planner进行直接角度控制
        io::Command command;
        if (target.has_value()) {
          std::list<auto_aim::Target> targets = {target.value()};
          command = aimer.aim(targets, std::chrono::steady_clock::now(), gs.bullet_speed);
        } else {
          command = {false, false, 0, 0};
        }
        
        auto is_empty = !target.has_value();

        // 发送简单的控制命令，不使用MPC的复杂控制量
        gimbal.send(
          command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0, is_empty);
        
        nlohmann::json data;
        
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["control"] = command.control;
        data["shoot"] = command.shoot;
        data["plan_yaw"] = command.yaw * 57.3;
        data["gimbal_yaw"] = gs.yaw * 57.3;
        data["plan_pitch"] = command.pitch * 57.3;
        data["gimbal_pitch"] = gs.pitch * 57.3;
        data["is_empty"] = is_empty;
        data["bullet_speed"] = gs.bullet_speed;
        data["robot_id"] = gs.id;

        plotter.plot(data);
        std::this_thread::sleep_for(10ms);
      } else
        std::this_thread::sleep_for(200ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      // 使用多线程YOLO检测器获取检测结果
      auto [debug_img, armors, debug_t] = detector.debug_pop();
      img = debug_img;  // 使用检测器返回的图像
      t = debug_t;

      auto q = gimbal.q(t);
      solver.set_R_gimbal2world(q);

      // 记录检测到的装甲板数量
      tools::logger()->info("Detected {} armors", armors.size());
      for (const auto& armor : armors) {
        tools::logger()->info("Armor center: ({}, {})", armor.center.x, armor.center.y);
      }
      
      auto targets = tracker.track(armors, t, gimbal.state().id);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      // 添加检测窗口显示
      cv::Mat display_img = img.clone();
      
      // 绘制装甲板边界框
      for (const auto &armor : armors) {
        if (armor.points.size() == 4) {
          // 绘制四边形边界框
          for (int i = 0; i < 4; i++) {
            cv::line(display_img, armor.points[i], armor.points[(i+1)%4], 
                    cv::Scalar(0, 255, 0), 2);
          }
          
          // 绘制装甲板中心点
          cv::circle(display_img, armor.center, 5, cv::Scalar(0, 0, 255), -1);
          
          // 显示装甲板ID和颜色
          std::string armor_info = fmt::format("ID:{}", static_cast<int>(armor.name));
          cv::putText(display_img, armor_info, armor.center + cv::Point2f(10, 0), 
                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
      }
      
      // 绘制画面中心点
      cv::circle(display_img, cv::Point(display_img.cols / 2, display_img.rows / 2), 
                5, cv::Scalar(255, 0, 0), -1);
      
      // 显示云台模式和装甲板数量
      std::string mode_text = fmt::format("Mode: {}", gimbal.str(mode));
      std::string armor_count_text = fmt::format("Armors: {}", armors.size());
      cv::putText(display_img, mode_text, cv::Point(10, 30), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
      cv::putText(display_img, armor_count_text, cv::Point(10, 60), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
      
      // 缩小显示图像
      cv::resize(display_img, display_img, cv::Size(), 0.5, 0.5);
      cv::imshow("YOLO Detection", display_img);
      
    } else {
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
      
      // 非自瞄模式时显示原始图像
      camera.read(img, t);
      cv::Mat display_img = img.clone();
      cv::resize(display_img, display_img, cv::Size(), 0.5, 0.5);
      cv::imshow("YOLO Detection", display_img);
    }
    
    auto key = cv::waitKey(1);
    if (key == 'q' || key == 27) break;  // q或ESC键退出
  }

  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  cv::destroyAllWindows();

  return 0;
}
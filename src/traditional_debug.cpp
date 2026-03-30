#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/detector.hpp"  // 使用传统检测器
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

  auto_aim::Detector detector(config_path, true);  // 使用传统检测器替代YOLO
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    auto last_fps_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);
        auto is_empty = target.has_value() ? false : true;

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc, is_empty);
        nlohmann::json data;
        
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["control"] = plan.control;
        data["fire"] = plan.fire;
        data["plan_yaw"] = plan.yaw * 57.3;
        data["gimbal_yaw"] = gs.yaw * 57.3;
        data["plan_pitch"] = plan.pitch * 57.3;
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

    camera.read(img, t);
    auto q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    //recorder.record(img, q, t);

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto armors = detector.detect(img);  // 使用传统检测方法
      
      
      auto targets = tracker.track(armors, t, gimbal.state().id);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    }
    
    else
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    
    
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  cv::destroyAllWindows();  // 关闭窗口

  return 0;
}
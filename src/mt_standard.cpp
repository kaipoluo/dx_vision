#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"  // 添加Gimbal头文件
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"  // 添加CommandGener头文件
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
//#include "tasks/auto_buff/buff_aimer.hpp"
//#include "tasks/auto_buff/buff_detector.hpp"
//#include "tasks/auto_buff/buff_solver.hpp"
//#include "tasks/auto_buff/buff_target.hpp"
//#include "tasks/auto_buff/buff_type.hpp"
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

// 将GimbalMode转换为io::Mode
io::Mode to_io_mode(io::GimbalMode gimbal_mode) {
  switch (gimbal_mode) {
    case io::GimbalMode::AUTO_AIM:
      return io::Mode::auto_aim;
    case io::GimbalMode::SMALL_BUFF:
      return io::Mode::small_buff;
    case io::GimbalMode::BIG_BUFF:
      return io::Mode::big_buff;
    case io::GimbalMode::IDLE:
    default:
      return io::Mode::idle;
  }
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);  

  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, gimbal, plotter,false);//debug开关
  //auto_buff::Buff_Detector buff_detector(config_path);
  //auto_buff::Solver buff_solver(config_path);
  //auto_buff::SmallTarget buff_small_target;
  //auto_buff::BigTarget buff_big_target;
  //auto_buff::Aimer buff_aimer(config_path);

  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode{io::Mode::idle};

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t);
      } else
        continue;
    }
  });

  while (!exiter.exit()) {
    // 从Gimbal获取当前模式
    auto gimbal_mode = gimbal.mode();
    mode = to_io_mode(gimbal_mode);

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode.load();
    }

    /// 自瞄
    if (mode.load() == io::Mode::auto_aim) {
      auto [img, armors, t] = detector.debug_pop();
      Eigen::Quaterniond q = gimbal.q(t - 1ms);  // 使用正确的q()方法获取四元数

      // recorder.record(img, q, t);

      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto targets = tracker.track(armors, t);
      
      commandgener.push(targets, t, gimbal.state().bullet_speed, ypr);  // 发送给决策线程
    }

    else
      continue;
  }

  detect_thread.join();// 等待自瞄线程结束

  return 0;
}
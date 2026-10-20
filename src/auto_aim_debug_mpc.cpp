#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/standard1.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter; //初始化两个对象
 
  cv::CommandLineParser cli(argc, argv, keys); //命令行解析工具
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) { //如果用户输入了help参数或没有指定配置文件路径，打印帮助信息并退出程序。
    cli.printMessage();
    return 0;
  }

//创建对象
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);


//线程安全队列 C++17中引入了std::optional类型，用于表示可能存在或不存在的值。这里使用std::optional<auto_aim::Target>来表示目标对象，可能存在也可能不存在。
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);  //模板参数 PopWhenFull = true：队列满时自动丢弃最旧数据（pop旧数据再push新数据）
  target_queue.push(std::nullopt);

  //atomic库：C++13中引入的库，用于线程安全的原子操作。这里使用std::atomic<bool>来表示一个布尔值，用于控制是否退出线程。
  std::atomic<bool> quit = false;
  std::atomic<bool> tracker_detecting{false};

  // ---- 延迟测量：主循环写，规划线程读 ----
  std::atomic<double> frame_ms{0}, detect_ms{0}, track_ms{0}, draw_ms{0}, plan_ms{0};
  std::atomic<long long> img_ns{0};  // 当前这张图的采集时刻（ns）

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();//chrono库：C++11中引入的库，用于处理时间和日期。这里使用std::chrono::steady_clock来获取当前时间点。
//steady_clock是一种单调时钟，适合用于测量时间间隔。std::chrono::steady_clock::now()返回一个表示当前时间点的对象。
  

//主循环    
while (!quit) {
      auto target = target_queue.front();//从队列中获取目标对象
      auto gs = gimbal.state(); //获取云台状态
      auto t_plan0 = std::chrono::steady_clock::now();
      auto plan = planner.plan(target, gs.bullet_speed);
      plan_ms = tools::delta_time(std::chrono::steady_clock::now(), t_plan0) * 1000.0;
      auto is_empty = target.has_value() ? false : true;

      float send_yaw = plan.yaw;

      // ！！解决yaw角度跳变问题，确保发给电控的yaw角度与当前云台yaw角度差值在[-π, π]范围内（解缠绕措施）
      while (send_yaw - gs.yaw > M_PI) send_yaw -= 2 * M_PI;
      while (send_yaw - gs.yaw < -M_PI) send_yaw += 2 * M_PI;

      float send_pitch = plan.pitch;
      while (send_pitch - gs.pitch > M_PI) send_pitch -= 2 * M_PI;
      while (send_pitch - gs.pitch < -M_PI) send_pitch += 2 * M_PI;

      gimbal.send(
        plan.control, plan.fire, send_yaw, plan.yaw_vel, plan.yaw_acc, send_pitch, plan.pitch_vel,
        plan.pitch_acc, is_empty);

      // ---- 延迟：从"这张图被采集"到"命令从串口发出" ----
      auto now = std::chrono::steady_clock::now();
      double latency_ms = 0;
      if (img_ns.load() > 0) {
        auto t_img = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(img_ns.load()));
        latency_ms = tools::delta_time(now, t_img) * 1000.0;
      }

      nlohmann::json data; //创建一个JSON对象，用于存储数据
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0); //delta_time函数：计算两个时间点之间的时间间隔 

      data["gimbal_yaw"] = gs.yaw * 57.2958;  // 世界坐标系中云台当前yaw角度（单位：度）


      data["yaw_rad"] = send_yaw;        // 发给电控的原始 yaw（弧度）
      data["pitch_rad"] = plan.pitch;    // 发给电控的原始 pitch（弧度）




      //  data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch * 57.2958;  // 世界坐标系中云台当前pitch角度（单位：度）
      //data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = plan.target_yaw * 57.2958;  //目标在世界坐标系中的yaw角度（单位：度）
      data["target_pitch"] =
        plan.target_pitch * 57.2958;  //目标在世界坐标系中的pitch角度（单位：度）

      data["plan_yaw"] = send_yaw * 57.2958;  // 期望在世界坐标系中的yaw角度（单位：度）
      // data["plan_yaw_vel"] = plan.yaw_vel;
      // data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch * 57.2958;  // 期望在世界坐标系中的pitch角度（单位：度）
      // data["plan_pitch_vel"] = plan.pitch_vel;
      // data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["is_empty"] = is_empty ? 1 : 0;
      data["bullet_speed"] = gs.bullet_speed;
      data["robot_id"] = gs.id;
      data["mode"] = plan.control ? (plan.fire ? 2 : 1) : 0;
      data["fire_deviation"] = planner.debug_fire_deviation;
      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z目标在世界坐标系中的z轴位置坐标
//        data["target_vz"] = target->ekf_x()[5];  //vz目标在z轴方向的速度分量
      }

      if (target.has_value()) {//has_value()函数：检查std::optional对象是否包含有效值。这里用于判断目标对象是否存在。
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      float diff_yaw = target.has_value() ? (send_yaw - gs.yaw) * 57.2958 : 0.0f;
      float diff_pitch = target.has_value() ? (plan.pitch - gs.pitch) * 57.2958 : 0.0f;
      data["diff_yaw"] = diff_yaw;
      data["diff_pitch"] = diff_pitch;
      data["distance"] = target.has_value() ? std::hypot(target->ekf_x()[0], target->ekf_x()[2]) : 0.0; //hypot函数：计算两个数的平方和的平方根，这里用于计算目标在世界坐标系中的水平距离（x轴和z轴的平方和的平方根）

      plotter.plot(data);

      static int print_cnt = 0;
      if (++print_cnt % 30 == 0) {
        tools::logger()->info(
          "[latency] 图像->发出 {:.1f} ms | 帧间隔 {:.1f} ms | 识别 {:.1f} ms | 跟踪 {:.1f} ms | "
          "画图 {:.1f} ms | 规划 {:.1f} ms",
          latency_ms, frame_ms.load(), detect_ms.load(), track_ms.load(), draw_ms.load(),
          plan_ms.load());
      }

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t; 

  // 帧率计算相关变量
  int frame_count = 0;
  double fps = 0.0;
  auto fps_start_time = std::chrono::steady_clock::now();

  auto last_read_t = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    camera.read(img, t);
    auto t_read = std::chrono::steady_clock::now();
    frame_ms = tools::delta_time(t_read, last_read_t) * 1000.0;
    last_read_t = t_read;
    img_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();

    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);//设置从云台坐标系到世界坐标系的旋转矩阵
    auto t_detect0 = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img);//使用YOLO模型检测图像中的装甲板，返回一个装甲板列表
    auto t_detect1 = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t,gimbal.state().id);//使用跟踪器对检测到的装甲板进行跟踪，返回一个目标列表
    auto t_track1 = std::chrono::steady_clock::now();
    detect_ms = tools::delta_time(t_detect1, t_detect0) * 1000.0;
    track_ms = tools::delta_time(t_track1, t_detect1) * 1000.0;
    auto t_draw0 = std::chrono::steady_clock::now();
    tracker_detecting = (tracker.state() == "detecting");//更新跟踪器状态，判断是否处于检测状态
    if (!targets.empty())
      target_queue.push(targets.front());
    else if (targets.empty())
      target_queue.push(std::nullopt);//如果没有检测到目标，则将std::nullopt推入队列，表示没有目标

    if (!targets.empty()) {//绘制目标投影
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list(); //获取目标的装甲板在世界坐标系中的位置和朝向信息，返回一个包含四维向量的列表，每个向量表示一个装甲板的x、y、z坐标和朝向角度。
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);//将装甲板在世界坐标系中的位置和朝向信息投影到图像中
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    // 计算帧率
    frame_count++;
    auto current_time = std::chrono::steady_clock::now();
    double elapsed_time = std::chrono::duration<double>(current_time - fps_start_time).count();
    
    if (elapsed_time >= 1.0) {  // 每秒更新一次帧率
      fps = frame_count / elapsed_time;
      frame_count = 0;
      fps_start_time = current_time;
    }

    // 在图像上显示帧率
    std::string fps_text = fmt::format("FPS: {:.2f}", fps);
    cv::putText(img, fps_text, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    draw_ms = tools::delta_time(std::chrono::steady_clock::now(), t_draw0) * 1000.0;
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);

  return 0;
}
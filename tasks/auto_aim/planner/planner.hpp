#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.001;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

// 云台实际指向（来自电控回传），用于“瞄准了才开火”的判定
struct GimbalInfo
{
  double yaw = 0;      // rad，世界系
  double pitch = 0;    // rad
  double yaw_vel = 0;  // rad/s
  bool valid = false;  // 没有云台反馈时置 false，此时不开火
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  double debug_fire_deviation = 0.0;
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed, const GimbalInfo & gimbal = {});
  Plan plan(std::optional<Target> target, double bullet_speed, const GimbalInfo & gimbal = {});

// 在Planner类的private部分添加基础偏移量成员变量
private:
  // 距离相关的偏移量配置，格式：distance -> (yaw_offset, pitch_offset)
  std::vector<std::tuple<double, double, double>> distance_offsets_;
  // 基础偏移量配置（当没有距离相关配置时使用）
  double yaw_offset_;
  double pitch_offset_;
  
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;

  // ---- 开火判定参数（“瞄准了才开火”）----
  // 允许的指向误差 = max(fire_k * 装甲板半宽 / 距离, fire_thresh)
  double fire_k_ = 1.5;           // 命中系数：1.0 = 刚好半个装甲板宽
  double fire_hysteresis_ = 1.5;  // 停火门限 = 开火门限 × 该值（防抖）
  double fire_min_frames_ = 2;    // 连续满足几个规划周期才开火
  double fire_yaw_vel_limit_ = 0.5;  // 云台角速度超过它就不开火 (rad/s)
  double fire_pitch_sign_ = 1.0;     // pitch 方向相反时改为 -1
  int fire_ok_count_ = 0;
  bool firing_ = false;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);
  
  // 根据距离计算插值后的偏移量
  std::pair<double, double> get_offset_by_distance(double distance);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
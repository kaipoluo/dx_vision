#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
// 在构造函数中添加基础偏移量的读取
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  
  // 读取基础偏移量配置，并转换为弧度
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  
  // 读取距离相关的偏移量配置
  if (yaml["distance_offsets"]) {
    auto offset_list = tools::read<std::vector<std::vector<double>>>(yaml, "distance_offsets");
    for (const auto & offset : offset_list) {
      if (offset.size() == 3) {
        distance_offsets_.emplace_back(offset[0], offset[1] / 57.3, offset[2] / 57.3);
      }
    }
  }
  
  // 其他构造函数代码保持不变
  // 开火判定参数：旧配置里没有的键用默认值，避免别的 yaml 起不来
  auto opt = [&](const char * key, double dflt) {
    return yaml[key] ? yaml[key].as<double>() : dflt;
  };
  fire_thresh_ = opt("fire_thresh", 0.01);  // 允许误差的绝对下限(rad)
  fire_k_ = opt("fire_k", 1.5);             // 命中系数
  fire_hysteresis_ = opt("fire_hysteresis", 1.5);
  fire_min_frames_ = opt("fire_min_frames", 2);
  fire_yaw_vel_limit_ = opt("fire_yaw_vel_limit", 0.5);
  fire_pitch_sign_ = opt("fire_pitch_sign", 1.0);
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed, const GimbalInfo & gimbal)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;//初始化最小距离为一个很大的数，确保在后续比较中能够找到实际的最小距离
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    debug_fire_deviation = 0.0;
    firing_ = false;
    fire_ok_count_ = 0;
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  debug_fire_deviation = std::hypot(
    traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
    traj(2, HALF_HORIZON + shoot_offset_) -
      pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_));

  // ---- 开火判定：云台“实际指向” 与 “预测瞄准点” 的夹角 ----
  // 允许误差 = 装甲板半宽 / 距离（乘命中系数），即目标在画面上有多大，就允许偏多少
  double half_width = (target.armor_type == ArmorType::big) ? 0.115 : 0.0675;
  double allow = std::max(fire_k_ * half_width / std::max(min_dist, 0.5), fire_thresh_);

  double aim_err = 1e9;
  bool vel_ok = false;
  if (gimbal.valid) {
    double e_yaw = tools::limit_rad(gimbal.yaw - plan.target_yaw);
    double e_pitch = fire_pitch_sign_ * (gimbal.pitch - plan.target_pitch);
    aim_err = std::hypot(e_yaw, e_pitch);
    vel_ok = std::abs(gimbal.yaw_vel) < fire_yaw_vel_limit_;
  }

  bool aligned = gimbal.valid && vel_ok && aim_err < allow;
  if (firing_) {
    if (!gimbal.valid || !vel_ok || aim_err > allow * fire_hysteresis_) firing_ = false;
  } else if (aligned) {
    if (++fire_ok_count_ >= static_cast<int>(fire_min_frames_)) {
      firing_ = true;
      fire_ok_count_ = 0;
    }
  } else {
    fire_ok_count_ = 0;
  }
  plan.fire = firing_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed, const GimbalInfo & gimbal)
{
  if (!target.has_value()) {
    debug_fire_deviation = 0.0;
    firing_ = false;
    fire_ok_count_ = 0;
    return {false};
  }

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed, gimbal);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 50;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 50;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;//初始化最小距离为一个很大的数，确保在后续比较中能够找到实际的最小距离

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  
  // 获取当前距离对应的偏移量
  auto [yaw_offset, pitch_offset] = get_offset_by_distance(min_dist);
  
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset), -bullet_traj.pitch - pitch_offset};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

std::pair<double, double> Planner::get_offset_by_distance(double distance)
{
  if (distance_offsets_.empty()) {
    return {yaw_offset_, pitch_offset_};
  }

  double dist_yaw, dist_pitch;

  if (distance <= std::get<0>(distance_offsets_.front())) {
    dist_yaw = std::get<1>(distance_offsets_.front());
    dist_pitch = std::get<2>(distance_offsets_.front());
  } else if (distance >= std::get<0>(distance_offsets_.back())) {
    dist_yaw = std::get<1>(distance_offsets_.back());
    dist_pitch = std::get<2>(distance_offsets_.back());
  } else {
    for (size_t i = 0; i < distance_offsets_.size() - 1; ++i) {
      double dist1 = std::get<0>(distance_offsets_[i]);
      double yaw1 = std::get<1>(distance_offsets_[i]);
      double pitch1 = std::get<2>(distance_offsets_[i]);

      double dist2 = std::get<0>(distance_offsets_[i + 1]);
      double yaw2 = std::get<1>(distance_offsets_[i + 1]);
      double pitch2 = std::get<2>(distance_offsets_[i + 1]);

      if (distance >= dist1 && distance <= dist2) {
        double ratio = (distance - dist1) / (dist2 - dist1);
        dist_yaw = yaw1 + ratio * (yaw2 - yaw1);
        dist_pitch = pitch1 + ratio * (pitch2 - pitch1);
        break;
      }
    }
  }

  return {yaw_offset_ + dist_yaw, pitch_offset_ + dist_pitch};
}

}  // namespace auto_aim
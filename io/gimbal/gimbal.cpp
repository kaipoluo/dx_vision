#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  auto baud_rate = tools::read<int>(yaml, "baud_rate");
  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(baud_rate); 
    serial::Timeout timeout = serial::Timeout::simpleTimeout(100);
    serial_.setTimeout(timeout);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  tools::logger()->info("[Gimbal] Initialized successfully.");
  // 初始化data_ahead_和data_behind_
  if (!queue_.empty()) {
  queue_.pop(data_ahead_);
  if (!queue_.empty()) {
    queue_.pop(data_behind_);
    queue_.push(data_behind_); // 将data_behind_放回队列
  } else {
    data_behind_ = data_ahead_;
  }
  queue_.push(data_ahead_); // 将data_ahead_放回队列
}
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  if (queue_.empty()) {
    // 队列空，返回当前状态的四元数
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::Quaterniond q = Eigen::AngleAxisd(state_.yaw, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(state_.pitch, Eigen::Vector3d::UnitY());
    return q.normalized();
  }

  // 查找最接近的两个数据点
  std::vector<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>> temp_queue;
  std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point> q_a, q_b;
  bool found = false;

  // 临时存储队列数据以便查找
  while (!queue_.empty()) {
    auto data = queue_.pop();
    temp_queue.push_back(data);
    
    if (std::get<1>(data) <= t) {
      q_a = data;
    } else if (std::get<1>(data) > t && std::get<1>(q_a) <= t) {
      q_b = data;
      found = true;
      break;
    }
  }

  // 恢复队列数据
  for (auto & data : temp_queue) {
    queue_.push(data);
  }

  if (!found) {
    // 没有找到合适的两个点，返回最接近的点
    if (temp_queue.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      Eigen::Quaterniond q = Eigen::AngleAxisd(state_.yaw, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(state_.pitch, Eigen::Vector3d::UnitY());
      return q.normalized();
    }
    return std::get<0>(temp_queue.back());
  }

  // 计算时间差
  double t_ab = tools::delta_time(std::get<1>(q_b), std::get<1>(q_a));
  double t_ac = tools::delta_time(t, std::get<1>(q_a));
  double k = t_ac / t_ab;

  // 使用SLERP进行插值
  Eigen::Quaterniond q_c = std::get<0>(q_a).slerp(k, std::get<0>(q_b));
  return q_c.normalized();
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.shoot = VisionToGimbal.shoot;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
     tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    if (rx_data_.head[0] != 'S' || rx_data_.head[1] != 'P') {
      error_count++;  // 这里需要增加错误计数，否则帧头不匹配时不会增加计数
      continue;
    }

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    // 直接从packed结构体复制值到临时变量，避免绑定问题
    uint16_t temp_crc16 = rx_data_.crc16;
    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
      tools::logger()->debug("[Gimbal] CRC16 check failed.RX:{}", temp_crc16);
      error_count++;
      continue;
    }

    error_count = 0;
    // 直接从packed结构体复制值到临时变量
    float temp_yaw = rx_data_.yaw;
    float temp_pitch = rx_data_.pitch;
    float temp_yaw_vel = rx_data_.yaw_vel;
    float temp_pitch_vel = rx_data_.pitch_vel;
    float temp_bullet_speed = rx_data_.bullet_speed;
    uint8_t temp_mode = rx_data_.mode;
    
    // 创建四元数并存储到队列
    Eigen::Quaterniond q = Eigen::AngleAxisd(temp_yaw, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(temp_pitch, Eigen::Vector3d::UnitY());
    queue_.push(std::make_tuple(q.normalized(), t));
    
    std::lock_guard<std::mutex> lock(mutex_);

    // 使用临时变量赋值
    state_.yaw = temp_yaw;
    state_.yaw_vel = temp_yaw_vel;
    state_.pitch = temp_pitch;
    state_.pitch_vel = temp_pitch_vel;
    state_.bullet_speed = temp_bullet_speed;
    // 删除弹丸剩余量

    switch (temp_mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", temp_mode);
        break;
    }
  
    // // 打印接收云台信息
    // tools::logger()->debug("[Gimbal] Data received successfully. Mode: {}, Yaw: {:.2f}, Pitch: {:.2f},CRC: {}",
    //                        static_cast<int>(temp_mode), temp_yaw * 57.3, temp_pitch * 57.3, temp_crc16);
    
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io
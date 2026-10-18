#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <robot.hpp>

constexpr int S = 3;

struct ROI_info {
    bool ROI_selected = false;
    int last_class = 0;
    cv::Rect2f ROI_bbox;
    void clear() { ROI_selected = false; }
};

inline Eigen::Vector3d pc_to_pw(const Eigen::Vector3d& pc, const Eigen::Matrix3d& R_IW) {
    return R_IW.transpose() * pc;
}

inline Eigen::Vector3d pw_to_pc(const Eigen::Vector3d& pw, const Eigen::Matrix3d& R_IW) {
    return R_IW * pw;
}

inline void re_project_point(
    cv::Mat& img,
    const Eigen::Vector3d& pw,
    const Eigen::Matrix3d& R_IW,
    const cv::Scalar& color)
{
    Eigen::Vector3d pc = pw_to_pc(pw, R_IW);
    if (pc(2, 0) <= 0) return;
    double u = pc(0, 0) / pc(2, 0);
    double v = pc(1, 0) / pc(2, 0);
    cv::circle(img, cv::Point2f(static_cast<float>(u), static_cast<float>(v)),
               5, color, -1);
}

template <int StateDim>
class _Kalman {
public:
    using Matrix_xxd = Eigen::Matrix<double, StateDim, StateDim>;
    using Matrix_zxd = Eigen::Matrix<double, 1, StateDim>;
    using Matrix_zzd = Eigen::Matrix<double, 1, 1>;
    using Matrix_x1d = Eigen::Matrix<double, StateDim, 1>;

    _Kalman(
        const Matrix_xxd& A,
        const Matrix_zxd& H,
        const Matrix_xxd& R,
        const Matrix_zzd& Q,
        const Matrix_x1d& init,
        double t0)
        : A_(A)
        , H_(H)
        , R_(R)
        , Q_(Q)
        , x_(init)
        , P_(Matrix_xxd::Identity() * 1000.0)
        , last_t_(t0)
    {}

    Matrix_x1d update(const Matrix_zxd& z, double t) {
        double dt = t - last_t_;
        last_t_ = t;
        if (dt <= 0) dt = 1e-3;

        Matrix_xxd F = build_F(dt);

        x_ = A_ * F * x_;
        P_ = A_ * F * P_ * F.transpose() * A_.transpose() + R_;

        Matrix_zxd y = z - H_ * x_;
        Matrix_zzd S = H_ * P_ * H_.transpose() + Q_;
        Matrix_x1d K = P_ * H_.transpose() * S.inverse();

        x_ = x_ + K * y;
        Matrix_xxd I = Matrix_xxd::Identity();
        P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * Q_ * K.transpose();

        return x_;
    }

    void reset(double value, double t) {
        x_.setZero();
        x_(0, 0) = value;
        P_ = Matrix_xxd::Identity() * 1000.0;
        last_t_ = t;
    }

private:
    Matrix_xxd A_, R_, P_;
    Matrix_zxd H_;
    Matrix_zzd Q_;
    Matrix_x1d x_;
    double last_t_;

    Matrix_xxd build_F(double dt) const {
        Matrix_xxd F = Matrix_xxd::Identity();
        double factorial = 1.0;
        double dt_pow = dt;
        for (int d = 1; d < StateDim; ++d) {
            factorial *= d;
            for (int i = 0; i < StateDim - d; ++i) {
                F(i, i + d) = dt_pow / factorial;
            }
            dt_pow *= dt;
        }
        return F;
    }
};

class PredictorKalman {
public:
    PredictorKalman();
    bool predict(Detection_pack& data, RobotCmd& send, cv::Mat& im2show);

private:
    Eigen::Vector3d pnp_get_pc(const cv::Point2f p[4], int armor_number);

    _Kalman<S> kalman;
    ROI_info roi;
    cv::Mat R_CI_MAT, F_MAT, C_MAT;
    Eigen::Matrix3d R_CI, F, C;
};
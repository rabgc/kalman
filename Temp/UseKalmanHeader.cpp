#include "Kalman.hpp"


KalmanFilter::KalmanFilter(const Eigen::MatrixXd& F, const Eigen::MatrixXd& H,
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                           const Eigen::MatrixXd& P, const Eigen::VectorXd& x)
    : F_(F), H_(H), Q_(Q), R_(R), P_(P), x_(x) {}

void KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(P_.rows(), P_.cols()) - K * H_) * P_;
}


Eigen::VectorXd KalmanFilter::getState() const {
    return x_;
}

Eigen::MatrixXd KalmanFilter::getCovariance() const {
    return P_;
}

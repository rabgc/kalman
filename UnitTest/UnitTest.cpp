#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../../include/doctest.h"
#include "../Kalman.hpp"
#include <Eigen/Dense>

// Example: Test KalmanFilter initialization
TEST_CASE("KalmanFilter initializes state and covariance correctly")
{
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(1, 2);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.1;
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);

  KalmanFilter kf(S, F, Q, R, P, x);

  CHECK((kf.getState() - x).norm() == doctest::Approx(0.0));
  CHECK((kf.getCovariance() - P).norm() == doctest::Approx(0.0));
}

// Example: Test prediction step
TEST_CASE("KalmanFilter prediction step updates state")
{
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2, 2); // State transition
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.1;
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);

  KalmanFilter kf(S, F, Q, R, P, x);

  kf.predict();

  // For identity F and zero x, state should remain zero
  CHECK((kf.getState() - x).norm() == doctest::Approx(0.0));
  // Covariance should increase by Q
  CHECK((kf.getCovariance() - (P + Q)).norm() == doctest::Approx(0.0));
}

// Example: Test update step
TEST_CASE("KalmanFilter update step assimilates measurement")
{
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 0.1;
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);

  KalmanFilter kf(S, F, Q, R, P, x);

  Eigen::VectorXd y = Eigen::VectorXd::Ones(2); // measurement

  kf.update(y);

  // After update, state should move toward measurement
  CHECK((kf.getState() - y).norm() < (y.norm())); // Should be closer to y than zero
  // Covariance should decrease (be less than initial P)
  CHECK(kf.getCovariance().norm() < P.norm());
}
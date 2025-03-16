/* File:   Kalman.cpp
 * Author: Rick.Bennett
 * Created on March 16, 2025, 6:00 AM
 */


#include "Kalman.hpp"  

KalmanFilter::KalmanFilter(const Eigen::MatrixXd& F, const Eigen::MatrixXd& H,
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                           const Eigen::MatrixXd& P, const Eigen::VectorXd& x)
    : F_(F), H_(H), Q_(Q), R_(R), P_(P), x_(x) {}

void KalmanFilter::predict() {
        x_ = F_ * x_;                        // propagate the state
        P_ = F_ * P_ * F_.transpose() + Q_;  // propagate the covariance 
    }

void KalmanFilter::update(const Eigen::VectorXd& z) {

        Eigen::VectorXd y = z - H_ * x_;   // innovations 
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_; // innovations cov
        Eigen::MatrixXd K = P_ * H_.transpose() * \
        S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols())); // Kalman gain

        x_ = x_ + K * y;  // state update 
        
        // The traditional covariance update can be unstable  
        // P_ = (Eigen::MatrixXd::Identity(P_.rows(), P_.cols()) - K * H_) * P_;  
        
        // So we use Joseph's stabilized update (cf. Bierman, 1977)
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_.rows(), P_.cols());
        P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();
        
        // Ensure positive definiteness is preserved (cf. Bierman, 1977)
        P_ = (P_ + P_.transpose()) / 2.0;

    }
    
    Eigen::VectorXd KalmanFilter::getState() const {
        return x_;
    }


    Eigen::MatrixXd KalmanFilter::getCovariance() const {
        return P_;
    }


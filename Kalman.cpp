/* File:   Kalman.cpp
 * Author: Rick.Bennett
 * Created on March 16, 2025, 6:00 AM
 */


#include "Kalman.hpp"  

KalmanFilter::KalmanFilter(
    const Eigen::MatrixXd& S, 
    const Eigen::MatrixXd& F, 
    const Eigen::MatrixXd& Q, 
    const Eigen::MatrixXd& R, 
    const Eigen::MatrixXd& P, 
    const Eigen::VectorXd& x
) : S_(S), F_(F), Q_(Q), R_(R), P_(P), x_(x) {

    // Ensure matrix dimensions are consistent
    assert(S_.rows() == S_.cols() 
        && "State transition S must be square.");
    assert(S_.rows() == x_.size() 
        && "State transition S and state vector x need same dim");
    assert(P_.rows() == P_.cols() 
        && "Cov matrix P must be square");
    assert(P_.rows() == x_.size() 
        && "Cov P matrix and state vector x need same dim");
    assert(Q_.rows() == Q_.cols() 
        && "Process noise cov matrix Q must be square");
    assert(Q_.rows() == S_.rows() 
        && "Process noise cov Q matrix and state transion S need same dim");
    assert(F_.cols() == x_.size() 
        && "Features matrix F need same no. of cols as state vector x");
    assert(R_.rows() == R_.cols() 
        && "Obs err cov matrix R must be square.");
    assert(F_.rows() == R_.rows() 
        && "Features matrix F and obs err cov matrix R need same dim");
    }

void KalmanFilter::predict() {
        x_ = S_ * x_;                        // propagate the state
        P_ = S_ * P_ * S_.transpose() + Q_;  // propagate the covariance 
    }

void KalmanFilter::update(const Eigen::VectorXd& z) {

        // Innovations 
        Eigen::VectorXd y = z - F_ * x_;   

        // Innovations covariance
        Eigen::MatrixXd C = F_ * P_ * F_.transpose() + R_; 

        // Kalman gain
        Eigen::MatrixXd K = P_ * F_.transpose() * \
        C.ldlt().solve(Eigen::MatrixXd::Identity(C.rows(), C.cols())); 

        // Update the state estimate
        x_ = x_ + K * y;  
        
        // The traditional covariance update can be unstable  
        // P_ = (Eigen::MatrixXd::Identity(P_.rows(), P_.cols()) - K * F_) * P_;  
        // So we use Joseph's stabilized update (cf. Bierman, 1977)
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_.rows(), P_.cols());
        P_ = (I - K * F_) * P_ * (I - K * F_).transpose() + K * R_ * K.transpose();
        
        // Ensure positive definiteness is preserved (cf. Bierman, 1977)
        P_ = (P_ + P_.transpose()) / 2.0;

    }
    
    Eigen::VectorXd KalmanFilter::getState() const {
        return x_;
    }


    Eigen::MatrixXd KalmanFilter::getCovariance() const {
        return P_;
    }


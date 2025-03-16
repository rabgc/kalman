/*
* Disclaimers
*/

/* File:   Kalman.hpp
 * Author: Rick.Bennett
 * Created on March 16, 2025, 6:00 AM
 */

#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>

/**
 * @class KalmanFilter
 * @brief Implements a Kalman Filter for state vector estimation for the 
 * simplified yet common case in which the time step, state transition, and 
 * design matrices all remain constant over the experiment.  
 * 
 * The state vector can be multi-dimensional, and the state vectors can be 
 * stacked to mtrack ultile objects (stations, satellites, etc).
 *
 * The KalmanFilter class provides methods for predicting and updating a state 
 * vector with an associated covariance matrix using a given state transition 
 * model.
 * 
 */
class KalmanFilter {
public:
    /**
     * @brief Constructor for the KalmanFilter class.
     * @param F State transition matrix.
     * @param H Observation model.
     * @param Q Process noise covariance.
     * @param R Measurement noise covariance.
     * @param P A priori state covariance.
     * @param x A priori state estimate.
     */
    KalmanFilter(const Eigen::MatrixXd& F, const Eigen::MatrixXd& H,
                 const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                 const Eigen::MatrixXd& P, const Eigen::VectorXd& x);

    /**
     * @brief Perform prediction step of the Kalman filter.
     */
    void predict();

    /**
     * @brief Update the state estimate using a new measurement.
     * @param z The measurement vector.
     */
    void update(const Eigen::VectorXd& z);

    /**
     * @brief Returns the current state estimate.
     * @return The state vector.
     */
    Eigen::VectorXd getState() const;

    /**
     * @brief Returns the current covariance matrix.
     * @return The covariance matrix.
     */
    Eigen::MatrixXd getCovariance() const;

private:
    Eigen::MatrixXd F_, H_, Q_, R_, P_;
    Eigen::VectorXd x_;
};

#endif // KALMAN_FILTER_HPP

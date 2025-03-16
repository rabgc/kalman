// Test the Kalman Filter implementation with a simple example
#include "../Kalman.hpp"  
#include <iostream>
#include <Eigen/Dense>
#include <vector>


// Multipe stacked n-dimensional state vectors 
int main() {
    
    // dynamics 
    int physical_dimensions = 3; // e.g., x, y, z
    int system_order = 2; //  1 or 2 for first order or second order

    // system state 
    int state_dimensions = physical_dimensions * system_order; // e.g., 6 for 3D state 
    int num_targets = 2; // e.g., GRPs or SV CMs 
    int stacked_state_size = state_dimensions * num_targets;

    // tracking data
    int num_epochs = 10; 
    int num_obs_dimensions = 3; // e.g., x, y, z
    int num_obs_per_epoch = num_obs_dimensions * num_targets; // e.g., 6 for 3 obs dimensions and 2 targets
    int num_stacked_obs = num_epochs * num_obs_per_epoch; // e.g., 18 for 3 epochs and 6 obs per epoch 

    // constant unit sampling
    double dt = 2.0; 

    // State transition matrix
    Eigen::MatrixXd Feye = Eigen::MatrixXd::Identity(physical_dimensions, physical_dimensions); 
    Eigen::MatrixXd F(stacked_state_size, stacked_state_size);
    for (int i =0; i < num_targets; i++) {

        // the first row of each target 
        int idx = i * state_dimensions; // 0, 6, 12 

        // positions 
        F.block(idx, idx, physical_dimensions, physical_dimensions) = Feye;
        
        // apply rate terms 
        for (int j = 1; j < system_order; j++) {

            int jdx = idx + j * physical_dimensions; // 3, 6, 9
            F.block(idx, jdx, physical_dimensions, physical_dimensions) = std::pow(dt, j) / std::tgamma(j+1) * Feye;

            // the identity part 
            F.block(idx + j * physical_dimensions, idx + j * physical_dimensions, physical_dimensions, physical_dimensions) = Feye;
        }
    }
    std::cout << "State Transition Matrix F:\n" << F << std::endl;

    // Design matrix
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_obs_per_epoch, stacked_state_size);
    for (int i = 0; i < num_targets; i++) {
        int idx = i * state_dimensions;
        std::cout << "indices" << i * num_obs_dimensions << idx <<  ":\n";
        H(i * num_obs_dimensions, idx) = 1; // observe x
        H(i * num_obs_dimensions + 1, idx + 1) = 1; // observe y
        H(i * num_obs_dimensions + 2, idx + 2) = 1; // observe z
    }
    std::cout << "Design matrix:\n" << H << "\n";

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(stacked_state_size, stacked_state_size) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(num_obs_per_epoch, num_obs_per_epoch) * 0.1;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(stacked_state_size, stacked_state_size);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(stacked_state_size);
    KalmanFilter kf(F, H, Q, R, P, x);

    // Run the simulation
    Eigen::VectorXd x_true = Eigen::VectorXd::Random(stacked_state_size) * 10.0;    
    std::cout << "True a priori state:\n" << x_true << "\n";
    
    for (int iepoch = 0; iepoch < num_epochs; ++iepoch) {

        // propagate the true state
        x_true = F * x_true; 

        // Simulate noisy data
        Eigen::VectorXd noise = Eigen::VectorXd::Random(num_obs_per_epoch);
        std::cout << "Noise:\n" << noise << "\n";
        Eigen::VectorXd d = H * x_true + 0.1 * noise; 
        std::cout << "Simulated data:\n" << d << "\n";

        // Kalman filter prediction and update
        kf.predict();
        std::cout << "Predicted State:\n" << kf.getState() << "\n";
        std::cout << "Predicted Covariance:\n" << kf.getCovariance() << "\n";

        kf.update(d);

        // Report the results
        std::cout << "Final Updated State:\n" << kf.getState() << "\n";
        //std::cout << "Final Updated Covariance:\n" << kf.getCovariance() << "\n";
        std::cout << "True state:\n" << x_true << "\n";
        std::cout << "True state - estimated state:\n" << x_true - kf.getState() << "\n";
    }

    return 0; // best practice to return 0
} // end of main

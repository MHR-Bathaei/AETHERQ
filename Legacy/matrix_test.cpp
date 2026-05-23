#include <iostream>
#include <chrono>
#include <Eigen/Dense>

int main() {
    // 1. Exact Input Shape X0: 8 sensor nodes, 6 features each
    Eigen::Matrix<float, 8, 6> X0 = Eigen::Matrix<float, 8, 6>::Random();
    
    // 2. Exact Weight Shape W_gcn1: 64 hidden features, 6 input features
    Eigen::Matrix<float, 64, 6> W_gcn1 = Eigen::Matrix<float, 64, 6>::Random();
    
    // 3. Begin timing the exact transformation
    auto start = std::chrono::high_resolution_clock::now();
    
    // Z1 = X0 * W_gcn1^T  (Resulting shape: 8 x 64)
    Eigen::Matrix<float, 8, 64> Z1 = X0 * W_gcn1.transpose();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "=======================================" << std::endl;
    std::cout << "   AETHER-Q: LOCKED REWRITE BASELINE   " << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "X0 (Sensor Inputs) Shape:   " << X0.rows() << " x " << X0.cols() << std::endl;
    std::cout << "W_gcn1 (Weights) Shape:     " << W_gcn1.rows() << " x " << W_gcn1.cols() << std::endl;
    std::cout << "Z1 (Output Tensor) Shape:   " << Z1.rows() << " x " << Z1.cols() << std::endl;
    std::cout << "Layer 1 Computation Time:   " << duration << " microseconds" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    return 0;
}
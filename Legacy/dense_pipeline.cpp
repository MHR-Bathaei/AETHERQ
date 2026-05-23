#include <iostream>
#include <chrono>
#include <Eigen/Dense>

int main() {
    // ==========================================
    // 1. STATIC ARCHITECTURE INITIALIZATION
    // ==========================================
    
    // Graph Normalization Constant Matrix: A_norm = (1/8) * ones(8,8)
    Eigen::Matrix<float, 8, 8> A_norm = Eigen::Matrix<float, 8, 8>::Constant(0.125f);
    
    // Inputs: X0 shape [8, 6] (8 sensors, 6 features each)
    Eigen::Matrix<float, 8, 6> X0 = Eigen::Matrix<float, 8, 6>::Random();
    
    // Layer 1 Parameters (GCNConv 1: 6 -> 64)
    Eigen::Matrix<float, 64, 6> W_gcn1 = Eigen::Matrix<float, 64, 6>::Random();
    Eigen::Matrix<float, 1, 64> b_gcn1 = Eigen::Matrix<float, 1, 64>::Random();
    
    // Layer 2 Parameters (GCNConv 2: 64 -> 64)
    Eigen::Matrix<float, 64, 64> W_gcn2 = Eigen::Matrix<float, 64, 64>::Random();
    Eigen::Matrix<float, 1, 64> b_gcn2 = Eigen::Matrix<float, 1, 64>::Random();
    
    // Layer 3 Parameters (GCNConv 3: 64 -> 64)
    Eigen::Matrix<float, 64, 64> W_gcn3 = Eigen::Matrix<float, 64, 64>::Random();
    Eigen::Matrix<float, 1, 64> b_gcn3 = Eigen::Matrix<float, 1, 64>::Random();
    
    // Linear Head 1 Parameters (Linear 1: 64 -> 64)
    Eigen::Matrix<float, 64, 64> W_lin1 = Eigen::Matrix<float, 64, 64>::Random();
    Eigen::Matrix<float, 1, 64> b_lin1 = Eigen::Matrix<float, 1, 64>::Random();
    
    // Linear Head 2 Parameters (Linear 2: 64 -> 3)
    Eigen::Matrix<float, 3, 64> W_lin2 = Eigen::Matrix<float, 3, 64>::Random();
    Eigen::Matrix<float, 1, 3>  b_lin2 = Eigen::Matrix<float, 1, 3>::Random();

    // ==========================================
    // 2. TIMED FORWARD PASS EXECUTION
    // ==========================================
    auto start = std::chrono::high_resolution_clock::now();
    
    // GCN Layer 1: Z1 = X0 @ W_gcn1.T -> Y1 = A_norm @ Z1 + b_gcn1 -> H1 = ReLU(Y1)
    Eigen::Matrix<float, 8, 64> Z1 = X0 * W_gcn1.transpose();
    Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
    
    // GCN Layer 2: Z2 = H1 @ W_gcn2.T -> Y2 = A_norm @ Z2 + b_gcn2 -> H2 = ReLU(Y2)
    Eigen::Matrix<float, 8, 64> Z2 = H1 * W_gcn2.transpose();
    Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
    
    // GCN Layer 3: Z3 = H2 @ W_gcn3.T -> Y3 = A_norm @ Z3 + b_gcn3 -> H3 = ReLU(Y3)
    Eigen::Matrix<float, 8, 64> Z3 = H2 * W_gcn3.transpose();
    Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
    
    // Linear Head Layer 1: L1 = ReLU(H3 @ W_lin1.T + b_lin1)
    Eigen::Matrix<float, 8, 64> L1 = (H3 * W_lin1.transpose() + b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
    
    // Linear Head Layer 2 (Node Predictions): NodePred = L1 @ W_lin2.T + b_lin2
    Eigen::Matrix<float, 8, 3> NodePred = L1 * W_lin2.transpose() + b_lin2.replicate<8, 1>();
    
    // Global Mean Pooling: Output = mean(NodePred, dim=0)
    Eigen::Matrix<float, 1, 3> Output = NodePred.colwise().mean();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ==========================================
    // 3. PERFORMANCE & ARCHITECTURE VERIFICATION
    // ==========================================
    std::cout << "=======================================" << std::endl;
    std::cout << "    AETHER-Q: FULL DENSE PIPELINE      " << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Pipeline Layout: 3xGCN -> 2xLinear -> MeanPool" << std::endl;
    std::cout << "Node Predictions Size:  " << NodePred.rows() << " x " << NodePred.cols() << std::endl;
    std::cout << "Final Output Vector Size: " << Output.rows() << " x " << Output.cols() << std::endl;
    std::cout << "Total Inference Time:     " << duration << " microseconds" << std::endl;
    std::cout << "Denoised Output Vector:   [" << Output(0,0) << ", " << Output(0,1) << ", " << Output(0,2) << "]" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    return 0;
}
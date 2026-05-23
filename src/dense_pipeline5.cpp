#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "model_weights.h" // <--- THE PYTORCH BRAIN INJECTION

// ==========================================
// MAXWELL PHYSICAL CONSTRAINT ENGINE
// ==========================================
float calculate_physics_confidence(const Eigen::Matrix<float, 8, 3>& B_pred, const Eigen::Matrix<float, 8, 6>& X_input) {
    Eigen::Matrix<float, 8, 3> positions = X_input.block<8, 3>(0, 3);
    Eigen::Matrix<float, 1, 3> centroid = positions.colwise().mean();
    float divergence_error = 0.0f;
    
    for (int i = 0; i < 8; ++i) {
        Eigen::Matrix<float, 1, 3> r_vec = positions.row(i) - centroid;
        float r_squared = r_vec.squaredNorm();
        if (r_squared > 1e-6f) { 
            float dot_prod = B_pred.row(i).dot(r_vec);
            divergence_error += std::abs(dot_prod / r_squared);
        }
    }
    return divergence_error / 8.0f;
}

int main() {
    // ==========================================
    // 1. MAPPING STATIC WEIGHTS (ZERO-COPY)
    // Python saves arrays as Row-Major, so we must explicitly tell Eigen to read them row-by-row.
    // ==========================================
    Eigen::Map<const Eigen::Matrix<float, 64, 6, Eigen::RowMajor>> e_W_gcn1(W_gcn1);
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn1(b_gcn1);
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_gcn2(W_gcn2);
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn2(b_gcn2);
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_gcn3(W_gcn3);
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn3(b_gcn3);
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_lin1(W_lin1);
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_lin1(b_lin1);
    
    Eigen::Map<const Eigen::Matrix<float, 3, 64, Eigen::RowMajor>> e_W_lin2(W_lin2);
    Eigen::Map<const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>> e_b_lin2(b_lin2);

    // ==========================================
    // 2. INPUT DATA PREPARATION
    // ==========================================
    Eigen::Matrix<float, 8, 8> A_norm = Eigen::Matrix<float, 8, 8>::Constant(0.125f);
    
    // Simulate raw noisy magnetic field inputs (Random for test purposes)
    Eigen::Matrix<float, 8, 6> X0 = Eigen::Matrix<float, 8, 6>::Random();
    
    // OVERRIDE Columns 3, 4, and 5 with EXACT physical sensor coordinates (0.10m spacing)
    // This gives the Maxwell engine actual spatial reality to evaluate.
    X0.row(0).tail<3>() << -0.05f, -0.05f, -0.05f;
    X0.row(1).tail<3>() << -0.05f, -0.05f,  0.05f;
    X0.row(2).tail<3>() << -0.05f,  0.05f, -0.05f;
    X0.row(3).tail<3>() << -0.05f,  0.05f,  0.05f;
    X0.row(4).tail<3>() <<  0.05f, -0.05f, -0.05f;
    X0.row(5).tail<3>() <<  0.05f, -0.05f,  0.05f;
    X0.row(6).tail<3>() <<  0.05f,  0.05f, -0.05f;
    X0.row(7).tail<3>() <<  0.05f,  0.05f,  0.05f;

    // ==========================================
    // 3. THE REAL FORWARD PASS
    // ==========================================
    auto start = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix<float, 8, 64> Z1 = X0 * e_W_gcn1.transpose();
    Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
    
    Eigen::Matrix<float, 8, 64> Z2 = H1 * e_W_gcn2.transpose();
    Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
    
    Eigen::Matrix<float, 8, 64> Z3 = H2 * e_W_gcn3.transpose();
    Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
    
    Eigen::Matrix<float, 8, 64> L1 = (H3 * e_W_lin1.transpose() + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
    Eigen::Matrix<float, 8, 3> NodePred = L1 * e_W_lin2.transpose() + e_b_lin2.replicate<8, 1>();
    
    Eigen::Matrix<float, 1, 3> Output = NodePred.colwise().mean();
    
    float physics_error = calculate_physics_confidence(NodePred, X0);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ==========================================
    // 4. FINAL TELEMETRY REPORT
    // ==========================================
    std::cout << "=======================================" << std::endl;
    std::cout << "   AETHER-Q: FULL DEPLOYMENT ENGINE    " << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Weights Loaded:           [SUCCESS] Zero-Copy Map" << std::endl;
    std::cout << "Sensor Array Layout:      [LOCKED] 0.10m Spacing" << std::endl;
    std::cout << "Total Inference Time:     " << duration << " microseconds" << std::endl;
    std::cout << "Denoised B-Field Output:  [" << Output(0,0) << ", " << Output(0,1) << ", " << Output(0,2) << "]" << std::endl;
    std::cout << "Maxwell Divergence Error: " << physics_error << std::endl;
    std::cout << "=======================================" << std::endl;
    
    return 0;
}
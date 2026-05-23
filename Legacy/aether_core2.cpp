#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "model_weights.h"

// =====================================================================
// AETHER-Q EDGE INFERENCE PIPELINE
// Production Object-Oriented Architecture
// =====================================================================
class AetherPipeline {
private:
    Eigen::Map<const Eigen::Matrix<float, 64, 6, Eigen::RowMajor>> e_W_gcn1;
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn1;
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_gcn2;
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn2;
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_gcn3;
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_gcn3;
    
    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> e_W_lin1;
    Eigen::Map<const Eigen::Matrix<float, 1, 64, Eigen::RowMajor>> e_b_lin1;
    
    Eigen::Map<const Eigen::Matrix<float, 3, 64, Eigen::RowMajor>> e_W_lin2;
    Eigen::Map<const Eigen::Matrix<float, 1, 3, Eigen::RowMajor>> e_b_lin2;

    Eigen::Matrix<float, 8, 8> A_norm;

public:
    AetherPipeline() : 
        e_W_gcn1(W_gcn1), e_b_gcn1(b_gcn1),
        e_W_gcn2(W_gcn2), e_b_gcn2(b_gcn2),
        e_W_gcn3(W_gcn3), e_b_gcn3(b_gcn3),
        e_W_lin1(W_lin1), e_b_lin1(b_lin1),
        e_W_lin2(W_lin2), e_b_lin2(b_lin2),
        A_norm(Eigen::Matrix<float, 8, 8>::Constant(0.125f)) {}

    Eigen::Matrix<float, 1, 3> predict(const Eigen::Matrix<float, 8, 6>& X0, Eigen::Matrix<float, 8, 3>& NodePred_out) {
        Eigen::Matrix<float, 8, 64> Z1 = X0 * e_W_gcn1.transpose();
        Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
        
        Eigen::Matrix<float, 8, 64> Z2 = H1 * e_W_gcn2.transpose();
        Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
        
        Eigen::Matrix<float, 8, 64> Z3 = H2 * e_W_gcn3.transpose();
        Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
        
        Eigen::Matrix<float, 8, 64> L1 = (H3 * e_W_lin1.transpose() + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
        NodePred_out = L1 * e_W_lin2.transpose() + e_b_lin2.replicate<8, 1>();
        
        return NodePred_out.colwise().mean();
    }

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
};

// =====================================================================
// FLIGHT TELEMETRY STREAM LOOP
// =====================================================================
int main() {
    AetherPipeline engine;
    
    std::ifstream infile("trajectory.csv");
    if (!infile.is_open()) {
        std::cerr << "[CRITICAL] Could not locate telemetry timeline: trajectory.csv" << std::endl;
        return 1;
    }

    std::cout << "=======================================" << std::endl;
    std::cout << "  AETHER-Q: FLIGHT TELEMETRY STREAMING " << std::endl;
    std::cout << "=======================================" << std::endl;

    std::string line;
    int timestep = 0;
    
    // Process flight frames sequentially
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string val;
        Eigen::Matrix<float, 8, 6> X0;
        
        // Parse 48 elements out of the CSV into the 8x6 matrix row-by-row
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (std::getline(ss, val, ',')) {
                    X0(i, j) = std::stof(val);
                }
            }
        }

        Eigen::Matrix<float, 8, 3> NodePred;
        
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<float, 1, 3> Final_B_Field = engine.predict(X0, NodePred);
        float physics_error = engine.calculate_physics_confidence(NodePred, X0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Print telemetry summary frame-by-frame
        std::cout << "T+" << timestep << " | DT: " << duration << "us | Error: " << physics_error 
                  << " | B-Field: [" << Final_B_Field(0,0) << ", " << Final_B_Field(0,1) << ", " << Final_B_Field(0,2) << "]" << std::endl;
        
        timestep++;
    }
    
    std::cout << "=======================================" << std::endl;
    std::cout << "[SUCCESS] Telemetry Stream Run Finished." << std::endl;
    return 0;
}
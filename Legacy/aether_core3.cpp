#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "model_weights.h"

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

    // -----------------------------------------------------------------
    // SILICON EMULATION: INT8 FIXED-POINT COMPRESSION KERNEL
    // -----------------------------------------------------------------
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> emulate_int8_matmul(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& X,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& W,
        float scale_factor) 
    {
        // 1. Scale and cast to 8-bit integer space (-128 to 127)
        auto X_int = (X / scale_factor).unaryExpr([](float val) {
            return std::max(-128.0f, std::min(127.0f, std::round(val)));
        });
        
        auto W_int = (W / scale_factor).unaryExpr([](float val) {
            return std::max(-128.0f, std::min(127.0f, std::round(val)));
        });

        // 2. Perform raw integer matrix multiplication (simulate FPGA DSP block execution)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> IntResult = X_int * W_int.transpose();

        // 3. Rescale back to float domains for downstream network layers
        return IntResult * (scale_factor * scale_factor);
    }

public:
    AetherPipeline() : 
        e_W_gcn1(W_gcn1), e_b_gcn1(b_gcn1),
        e_W_gcn2(W_gcn2), e_b_gcn2(b_gcn2),
        e_W_gcn3(W_gcn3), e_b_gcn3(b_gcn3),
        e_W_lin1(W_lin1), e_b_lin1(b_lin1),
        e_W_lin2(W_lin2), e_b_lin2(b_lin2),
        A_norm(Eigen::Matrix<float, 8, 8>::Constant(0.125f)) {}

    // Quantized Predict Optimization Loop
    Eigen::Matrix<float, 1, 3> predict_quantized(const Eigen::Matrix<float, 8, 6>& X0, Eigen::Matrix<float, 8, 3>& NodePred_out) {
        // Define quantization constant based on maximum weight ranges (dynamic scaling)
        const float SF = 0.0078f; // 1/128 scaling step for standard normalized weights

        // Layer 1 GCN Matrix Ops running through our simulated integer hardware kernel
        Eigen::Matrix<float, 8, 64> Z1 = emulate_int8_matmul(X0, e_W_gcn1, SF);
        Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
        
        // Layer 2 GCN Matrix Ops
        Eigen::Matrix<float, 8, 64> Z2 = emulate_int8_matmul(H1, e_W_gcn2, SF);
        Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
        
        // Layer 3 GCN Matrix Ops
        Eigen::Matrix<float, 8, 64> Z3 = emulate_int8_matmul(H2, e_W_gcn3, SF);
        Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
        
        // Linear Layer 1 Dense Map
        Eigen::Matrix<float, 8, 64> L1 = (emulate_int8_matmul(H3, e_W_lin1, SF) + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
        
        // Terminal Regression Map
        NodePred_out = emulate_int8_matmul(L1, e_W_lin2, SF) + e_b_lin2.replicate<8, 1>();
        
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

int main() {
    AetherPipeline engine;
    
    std::ifstream infile("trajectory.csv");
    if (!infile.is_open()) {
        std::cerr << "[CRITICAL] Could not locate telemetry timeline: trajectory.csv" << std::endl;
        return 1;
    }

    std::cout << "=======================================" << std::endl;
    std::cout << " AETHER-Q: SILICON QUANTIZATION CORE   " << std::endl;
    std::cout << "=======================================" << std::endl;

    std::string line;
    int timestep = 0;
    
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string val;
        Eigen::Matrix<float, 8, 6> X0;
        
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (std::getline(ss, val, ',')) {
                    X0(i, j) = std::stof(val);
                }
            }
        }

        Eigen::Matrix<float, 8, 3> NodePred;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // EXECUTE QUANTIZED INFERENCE
        Eigen::Matrix<float, 1, 3> Final_B_Field = engine.predict_quantized(X0, NodePred);
        float physics_error = engine.calculate_physics_confidence(NodePred, X0);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "T+" << timestep << " | Q-DT: " << duration << "us | Physics Error: " << physics_error 
                  << " | B-Field: [" << Final_B_Field(0,0) << ", " << Final_B_Field(0,1) << ", " << Final_B_Field(0,2) << "]" << std::endl;
        
        timestep++;
    }
    
    std::cout << "=======================================" << std::endl;
    std::cout << "[SUCCESS] Quantized Stream Run Finished." << std::endl;
    return 0;
}
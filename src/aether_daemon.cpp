#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "model_weights.h"

// Define to 1 to run the INT8 emulated core, or 0 for FP32 native
#define COMPRESSION_MODE_INT8 0 

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
    const float SF = 0.0078f;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> emulate_int8_matmul(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& X,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& W) 
    {
        auto X_int = (X / SF).unaryExpr([](float val) {
            return std::max(-128.0f, std::min(127.0f, std::round(val)));
        });
        auto W_int = (W / SF).unaryExpr([](float val) {
            return std::max(-128.0f, std::min(127.0f, std::round(val)));
        });
        return (X_int * W_int.transpose()) * (SF * SF);
    }

public:
    AetherPipeline() : 
        e_W_gcn1(W_gcn1), e_b_gcn1(b_gcn1),
        e_W_gcn2(W_gcn2), e_b_gcn2(b_gcn2),
        e_W_gcn3(W_gcn3), e_b_gcn3(b_gcn3),
        e_W_lin1(W_lin1), e_b_lin1(b_lin1),
        e_W_lin2(W_lin2), e_b_lin2(b_lin2),
        A_norm(Eigen::Matrix<float, 8, 8>::Constant(0.125f)) {}

    Eigen::Matrix<float, 1, 3> predict(const Eigen::Matrix<float, 8, 6>& X0, Eigen::Matrix<float, 8, 3>& NodePred_out) {
        #if COMPRESSION_MODE_INT8
            Eigen::Matrix<float, 8, 64> Z1 = emulate_int8_matmul(X0, e_W_gcn1);
            Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z2 = emulate_int8_matmul(H1, e_W_gcn2);
            Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z3 = emulate_int8_matmul(H2, e_W_gcn3);
            Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> L1 = (emulate_int8_matmul(H3, e_W_lin1) + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
            NodePred_out = emulate_int8_matmul(L1, e_W_lin2) + e_b_lin2.replicate<8, 1>();
        #else
            Eigen::Matrix<float, 8, 64> Z1 = X0 * e_W_gcn1.transpose();
            Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z2 = H1 * e_W_gcn2.transpose();
            Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z3 = H2 * e_W_gcn3.transpose();
            Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> L1 = (H3 * e_W_lin1.transpose() + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
            NodePred_out = L1 * e_W_lin2.transpose() + e_b_lin2.replicate<8, 1>();
        #endif
        return NodePred_out.colwise().mean();
    }
};

int main() {
    AetherPipeline engine;
    std::string input_line;
    
    // Announce service readiness to the host OS
    std::cout << "[AETHER DAEMON] Online and listening for telemetry..." << std::endl;

    // Infinite listening loop (The Microservice Architecture)
    while (std::getline(std::cin, input_line)) {
        if (input_line == "EXIT") break;
        if (input_line.empty()) continue;

        std::stringstream ss(input_line);
        std::string val;
        Eigen::Matrix<float, 8, 6> X0;
        
        int count = 0;
        while (std::getline(ss, val, ',') && count < 48) {
            X0(count / 6, count % 6) = std::stof(val);
            count++;
        }

        if (count == 48) {
            Eigen::Matrix<float, 8, 3> NodePred;
            auto start = std::chrono::high_resolution_clock::now();
            
            Eigen::Matrix<float, 1, 3> B_Field = engine.predict(X0, NodePred);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // Output JSON-style response for other programs to consume
            std::cout << "{\"status\":\"success\", \"latency_us\":" << duration 
                      << ", \"B_field\":[" << B_Field(0,0) << "," << B_Field(0,1) << "," << B_Field(0,2) << "]}" << std::endl;
        } else {
            std::cout << "{\"status\":\"error\", \"message\":\"Malformed payload. Expected 48 float values.\"}" << std::endl;
        }
    }
    
    std::cout << "[AETHER DAEMON] Offline." << std::endl;
    return 0;
}
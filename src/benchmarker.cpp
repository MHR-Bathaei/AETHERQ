#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include "model_weights.h"

class ProfilerPipeline {
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
    ProfilerPipeline() : 
        e_W_gcn1(W_gcn1), e_b_gcn1(b_gcn1),
        e_W_gcn2(W_gcn2), e_b_gcn2(b_gcn2),
        e_W_gcn3(W_gcn3), e_b_gcn3(b_gcn3), 
        e_W_lin1(W_lin1), e_b_lin1(b_lin1),
        e_W_lin2(W_lin2), e_b_lin2(b_lin2),
        A_norm(Eigen::Matrix<float, 8, 8>::Constant(0.125f)) {}

    void run_inference(const Eigen::Matrix<float, 8, 6>& X0, bool use_int8, Eigen::Matrix<float, 8, 3>& NodePred_out) {
        if (use_int8) {
            Eigen::Matrix<float, 8, 64> Z1 = emulate_int8_matmul(X0, e_W_gcn1);
            Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z2 = emulate_int8_matmul(H1, e_W_gcn2);
            Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z3 = emulate_int8_matmul(H2, e_W_gcn3);
            Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> L1 = (emulate_int8_matmul(H3, e_W_lin1) + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
            NodePred_out = emulate_int8_matmul(L1, e_W_lin2) + e_b_lin2.replicate<8, 1>();
        } else {
            Eigen::Matrix<float, 8, 64> Z1 = X0 * e_W_gcn1.transpose();
            Eigen::Matrix<float, 8, 64> H1 = (A_norm * Z1 + e_b_gcn1.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z2 = H1 * e_W_gcn2.transpose();
            Eigen::Matrix<float, 8, 64> H2 = (A_norm * Z2 + e_b_gcn2.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> Z3 = H2 * e_W_gcn3.transpose();
            Eigen::Matrix<float, 8, 64> H3 = (A_norm * Z3 + e_b_gcn3.replicate<8, 1>()).cwiseMax(0.0f);
            Eigen::Matrix<float, 8, 64> L1 = (H3 * e_W_lin1.transpose() + e_b_lin1.replicate<8, 1>()).cwiseMax(0.0f);
            NodePred_out = L1 * e_W_lin2.transpose() + e_b_lin2.replicate<8, 1>();
        }
    }
};

struct RunStats {
    double min_us;
    double max_us;
    double mean_us;
    double p99_us;
};

RunStats compute_stats(std::vector<double>& latencies) {
    RunStats stats;
    stats.min_us = *std::min_element(latencies.begin(), latencies.end());
    stats.max_us = *std::max_element(latencies.begin(), latencies.end());
    
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    stats.mean_us = sum / latencies.size();
    
    std::sort(latencies.begin(), latencies.end());
    size_t p99_idx = static_cast<size_t>(latencies.size() * 0.99);
    stats.p99_us = latencies[p99_idx];
    
    return stats;
}

int main() {
    ProfilerPipeline engine;
    const int num_trials = 5000;
    
    Eigen::Matrix<float, 8, 6> X_mock = Eigen::Matrix<float, 8, 6>::Random();
    Eigen::Matrix<float, 8, 3> NodePred;

    std::vector<double> fp32_latencies;
    std::vector<double> int8_latencies;

    std::cout << "[STARTING] Profiling Core Pipeline over " << num_trials << " iterations..." << std::endl;

    for(int i = 0; i < num_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.run_inference(X_mock, false, NodePred);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
        fp32_latencies.push_back(duration);
    }

    for(int i = 0; i < num_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.run_inference(X_mock, true, NodePred);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
        int8_latencies.push_back(duration);
    }

    RunStats fp32_stats = compute_stats(fp32_latencies);
    RunStats int8_stats = compute_stats(int8_latencies);

    std::ofstream csv_file("benchmarks_output.csv");
    csv_file << "metric,fp32_latency_us,int8_emulated_latency_us\n";
    csv_file << "min," << fp32_stats.min_us << "," << int8_stats.min_us << "\n";
    csv_file << "max," << fp32_stats.max_us << "," << int8_stats.max_us << "\n";
    csv_file << "mean," << fp32_stats.mean_us << "," << int8_stats.mean_us << "\n";
    csv_file << "p99," << fp32_stats.p99_us << "," << int8_stats.p99_us << "\n";
    csv_file.close();

    std::cout << "\n=======================================" << std::endl;
    std::cout << "  AMD HARDWARE PROFILING SUITE METRICS " << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "FP32 Profile  -> Min: " << fp32_stats.min_us << "us | Mean: " << fp32_stats.mean_us << "us | P99: " << fp32_stats.p99_us << "us" << std::endl;
    std::cout << "INT8 Profile  -> Min: " << int8_stats.min_us << "us | Mean: " << int8_stats.mean_us << "us | P99: " << int8_stats.p99_us << "us" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "[SUCCESS] Profiling complete. Data locked into 'benchmarks_output.csv'." << std::endl;

    return 0;
}
// bayesian_optimizer.h — Batch T3: Bayesian optimization with GP surrogate + EI acquisition
#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

#include <sai/core/error.h>
#include <sai/tuning/tuning_objective.h>
#include <sai/tuning/tuning_space.h>

namespace sai::tuning {

struct OptimizerConfig {
    std::size_t max_iterations{50};
    std::size_t initial_random_points{5};
    double noise_level{0.001};
    double kernel_sigma2{1.0};
    double kernel_length{0.5};
    double jitter{1e-6};
    std::size_t lbfgs_restarts{10};
    std::size_t lbfgs_max_iters{100};
    double lbfgs_grad_tol{1e-5};
    double ei_tol{1e-6};
    std::size_t ei_stagnation_iters{3};
};

struct OptimizationPoint {
    std::vector<double> params;
    double cost{0.0};
    std::chrono::system_clock::time_point timestamp;
};

class BayesianOptimizer {
public:
    BayesianOptimizer(TuningSpace space, OptimizerConfig config);

    auto AddObservation(OptimizationPoint point) -> void;

    auto Optimize(ITuningObjective& objective,
                  std::chrono::system_clock::time_point since) -> Result<OptimizationPoint>;

    auto BestPoint() const -> const OptimizationPoint&;
    auto AllObservations() const -> const std::vector<OptimizationPoint>&;

private:
    struct PredictResult {
        double mean;
        double variance;
    };

    // GP kernel
    auto RBFKernel(const std::vector<double>& x, const std::vector<double>& y) const -> double;

    // Matrix operations
    auto BuildKernelMatrix() const -> std::vector<std::vector<double>>;
    auto CholeskyDecompose(std::vector<std::vector<double>>& K,
                           std::vector<std::vector<double>>& L) const -> bool;
    auto SolveCholesky(const std::vector<std::vector<double>>& L,
                       const std::vector<double>& b) const -> std::vector<double>;

    // GP prediction & acquisition
    auto Predict(const std::vector<double>& point) const -> PredictResult;
    auto ComputeEI(const std::vector<double>& point) const -> double;
    auto ComputeLogMarginalLikelihood(double log_sigma2, double log_length) const -> double;
    auto FitHyperparameters() -> void;

    // L-BFGS-B for EI maximization
    auto LBFGSBMaximize() -> std::vector<double>;
    auto ComputeEIGradient(const std::vector<double>& point) const -> std::vector<double>;
    static auto NormalCDF(double x) -> double;
    static auto NormalPDF(double x) -> double;

    // Utility
    auto RandomFeasiblePoint() const -> std::vector<double>;
    static auto SqDistance(const std::vector<double>& a,
                           const std::vector<double>& b) -> double;

    // Cache for fast GP predictions (validated by cache_valid_)
    auto EnsureCache() const -> void;

    TuningSpace space_;
    OptimizerConfig config_;
    std::vector<OptimizationPoint> observations_;

    // GP hyperparameters
    double sigma2_;
    double length_;

    // Cholesky cache — invalidated when observations change or hyperparams update
    mutable std::vector<std::vector<double>> cached_L_;
    mutable std::vector<double> cached_alpha_;
    mutable bool cache_valid_{false};
};

}  // namespace sai::tuning

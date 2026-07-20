// bayesian_optimizer.cpp — GP surrogate + EI acquisition + L-BFGS-B
#include <sai/tuning/bayesian_optimizer.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace sai::tuning {

// ── Constructor ─────────────────────────────────────────────────────

BayesianOptimizer::BayesianOptimizer(TuningSpace space, OptimizerConfig config)
    : space_(std::move(space))
    , config_(std::move(config))
    , sigma2_(config_.kernel_sigma2)
    , length_(config_.kernel_length) {}

// ── Observation management ──────────────────────────────────────────

auto BayesianOptimizer::AddObservation(OptimizationPoint point) -> void {
    observations_.push_back(std::move(point));
    cache_valid_ = false;
}

auto BayesianOptimizer::BestPoint() const -> const OptimizationPoint& {
    if (observations_.empty()) {
        // Return a reference to an invalid state — caller must ensure non-empty
        static const OptimizationPoint empty;
        return empty;
    }
    const OptimizationPoint* best = &observations_[0];
    for (const auto& obs : observations_) {
        if (obs.cost < best->cost) best = &obs;
    }
    return *best;
}

auto BayesianOptimizer::AllObservations() const -> const std::vector<OptimizationPoint>& {
    return observations_;
}

// ── GP kernel ──────────────────────────────────────────────────────

auto BayesianOptimizer::RBFKernel(const std::vector<double>& x,
                                   const std::vector<double>& y) const -> double {
    return sigma2_ * std::exp(-SqDistance(x, y) / (2.0 * length_ * length_));
}

auto BayesianOptimizer::SqDistance(const std::vector<double>& a,
                                    const std::vector<double>& b) -> double {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// ── Matrix operations ───────────────────────────────────────────────

auto BayesianOptimizer::BuildKernelMatrix() const -> std::vector<std::vector<double>> {
    std::size_t n = observations_.size();
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double val = RBFKernel(observations_[i].params, observations_[j].params);
            K[i][j] = val;
            K[j][i] = val;
        }
        // Add observation noise to diagonal
        K[i][i] += config_.noise_level;
    }
    return K;
}

auto BayesianOptimizer::CholeskyDecompose(std::vector<std::vector<double>>& K,
                                           std::vector<std::vector<double>>& L) const -> bool {
    std::size_t n = K.size();
    L.assign(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = K[i][j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }
            if (i == j) {
                if (sum <= 0.0) return false;
                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return true;
}

auto BayesianOptimizer::SolveCholesky(const std::vector<std::vector<double>>& L,
                                       const std::vector<double>& b) const -> std::vector<double> {
    std::size_t n = L.size();
    std::vector<double> x(n);

    // Forward substitution: L * y = b
    for (std::size_t i = 0; i < n; ++i) {
        double sum = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            sum -= L[i][j] * x[j];
        }
        x[i] = sum / L[i][i];
    }

    // Backward substitution: L^T * x = y (reuse x as result)
    for (std::size_t i = n; i-- > 0;) {
        double sum = x[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= L[j][i] * x[j];
        }
        x[i] = sum / L[i][i];
    }
    return x;
}

// ── GP prediction ──────────────────────────────────────────────────

auto BayesianOptimizer::EnsureCache() const -> void {
    if (cache_valid_) return;

    std::size_t n = observations_.size();
    if (n == 0) {
        cache_valid_ = true;
        return;
    }

    // Build kernel matrix with jitter for numerical stability
    double jitter = config_.jitter;
    for (int attempt = 0; attempt < 5; ++attempt) {
        auto K = BuildKernelMatrix();
        for (std::size_t i = 0; i < n; ++i) {
            K[i][i] += jitter;
        }
        if (CholeskyDecompose(K, cached_L_)) {
            // Compute alpha = K^{-1} * y
            std::vector<double> y(n);
            for (std::size_t i = 0; i < n; ++i) {
                y[i] = observations_[i].cost;
            }
            cached_alpha_ = SolveCholesky(cached_L_, y);
            cache_valid_ = true;
            return;
        }
        jitter *= 10.0;
    }
    // If all attempts fail, jitter is already large — one last try
    jitter *= 10.0;
    auto K = BuildKernelMatrix();
    for (std::size_t i = 0; i < n; ++i) {
        K[i][i] += jitter;
    }
    if (!CholeskyDecompose(K, cached_L_)) {
        // Matrix still non-PD despite maximal jitter — leave cache invalid;
        // Predict will return safe defaults when cache_valid_ is false.
        cache_valid_ = false;
        return;
    }
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = observations_[i].cost;
    }
    cached_alpha_ = SolveCholesky(cached_L_, y);
    cache_valid_ = true;
}

auto BayesianOptimizer::Predict(const std::vector<double>& point) const -> PredictResult {
    EnsureCache();

    std::size_t n = observations_.size();
    if (n == 0 || !cache_valid_) return {0.0, sigma2_};

    // Compute k* vector
    std::vector<double> kstar(n);
    for (std::size_t i = 0; i < n; ++i) {
        kstar[i] = RBFKernel(point, observations_[i].params);
    }

    // Solve K * beta = k* using cached Cholesky
    auto beta = SolveCholesky(cached_L_, kstar);

    // Mean: k*^T * alpha
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mean += kstar[i] * cached_alpha_[i];
    }

    // Variance: k(x*,x*) - k*^T * beta
    double kxx = sigma2_;
    double variance = kxx;
    for (std::size_t i = 0; i < n; ++i) {
        variance -= kstar[i] * beta[i];
    }

    return {mean, std::max(variance, 0.0)};
}

// ── Expected Improvement ──────────────────────────────────────────

auto BayesianOptimizer::NormalCDF(double x) -> double {
    return 0.5 * (1.0 + std::erf(x / std::numbers::sqrt2));
}

auto BayesianOptimizer::NormalPDF(double x) -> double {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
}

auto BayesianOptimizer::ComputeEI(const std::vector<double>& point) const -> double {
    if (observations_.empty()) return 1.0;

    auto [mean, variance] = Predict(point);
    double sigma = std::sqrt(variance);

    double f_best = BestPoint().cost;

    if (sigma < 1e-12) return 0.0;

    // Z = (f_best - mu) / sigma  — for minimization
    double Z = (f_best - mean) / sigma;
    return (f_best - mean) * NormalCDF(Z) + sigma * NormalPDF(Z);
}

auto BayesianOptimizer::ComputeEIGradient(const std::vector<double>& point) const -> std::vector<double> {
    std::size_t dim = point.size();
    std::vector<double> grad(dim, 0.0);
    double h = 1e-6;

    for (std::size_t i = 0; i < dim; ++i) {
        std::vector<double> x_plus = point;
        std::vector<double> x_minus = point;
        x_plus[i] += h;
        x_minus[i] -= h;

        // Clamp perturbed points to bounds
        space_.ClampToBounds(x_plus);
        space_.ClampToBounds(x_minus);

        grad[i] = (ComputeEI(x_plus) - ComputeEI(x_minus)) / (2.0 * h);
    }
    return grad;
}

// ── GP hyperparameter fitting ───────────────────────────────────────

auto BayesianOptimizer::ComputeLogMarginalLikelihood(double log_sigma2,
                                                       double log_length) const -> double {
    std::size_t n = observations_.size();
    if (n == 0) return 0.0;

    double s2_tmp = std::exp(log_sigma2);
    double l_tmp = std::exp(log_length);

    // Build kernel matrix with temporary hyperparameters
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double r = SqDistance(observations_[i].params, observations_[j].params);
            double val = s2_tmp * std::exp(-r / (2.0 * l_tmp * l_tmp));
            K[i][j] = val;
            K[j][i] = val;
        }
        K[i][i] += config_.noise_level + config_.jitter;
    }

    // Cholesky decomposition
    std::vector<std::vector<double>> L;
    if (!CholeskyDecompose(K, L)) {
        // Non-PD despite jitter — return very low likelihood
        return -1e10;
    }

    // alpha = K^{-1} * y
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = observations_[i].cost;
    }
    auto alpha = SolveCholesky(L, y);

    // log marginal likelihood = -0.5 * y^T * alpha - sum(log(L_ii)) - 0.5 * n * log(2*pi)
    double y_alpha = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        y_alpha += y[i] * alpha[i];
    }

    double log_det = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        log_det += std::log(L[i][i]);
    }

    return -0.5 * y_alpha - log_det - 0.5 * static_cast<double>(n) * std::log(2.0 * std::numbers::pi);
}

auto BayesianOptimizer::FitHyperparameters() -> void {
    std::size_t n = observations_.size();
    if (n < 2) return;

    double a = std::log(sigma2_);
    double b = std::log(length_);

    std::size_t max_iters_local = 50;
    double tol = 1e-4;

    double prev_lml = ComputeLogMarginalLikelihood(a, b);

    for (std::size_t iter = 0; iter < max_iters_local; ++iter) {
        double h = 1e-4;

        // Finite-difference gradients
        double lml_ap = ComputeLogMarginalLikelihood(a + h, b);
        double lml_am = ComputeLogMarginalLikelihood(a - h, b);
        double grad_a = (lml_ap - lml_am) / (2.0 * h);

        double lml_bp = ComputeLogMarginalLikelihood(a, b + h);
        double lml_bm = ComputeLogMarginalLikelihood(a, b - h);
        double grad_b = (lml_bp - lml_bm) / (2.0 * h);

        // Gradient ascent with line search (backtracking)
        double step = 1.0;
        double c = 1e-4;  // Armijo sufficient decrease parameter

        while (step > 1e-8) {
            double a_new = a + step * grad_a;
            double b_new = b + step * grad_b;
            double new_lml = ComputeLogMarginalLikelihood(a_new, b_new);

            // Armijo condition: L(new) > L(old) + c * step * ||grad||^2
            double grad_norm_sq = grad_a * grad_a + grad_b * grad_b;
            if (new_lml > prev_lml + c * step * grad_norm_sq) {
                a = a_new;
                b = b_new;
                break;
            }
            step *= 0.5;
        }

        double cur_lml = ComputeLogMarginalLikelihood(a, b);
        if (std::abs(cur_lml - prev_lml) < tol) break;
        prev_lml = cur_lml;
    }

    sigma2_ = std::exp(a);
    length_ = std::exp(b);
    cache_valid_ = false;
}

// ── L-BFGS-B for EI maximization ────────────────────────────────────

auto BayesianOptimizer::LBFGSBMaximize() -> std::vector<double> {
    std::size_t dim = space_.Dimension();
    if (dim == 0) return {};

    std::size_t m = 5;  // history size

    std::vector<double> best_x;
    double best_ei = -std::numeric_limits<double>::infinity();

    for (std::size_t restart = 0; restart < config_.lbfgs_restarts; ++restart) {
        std::vector<double> x = RandomFeasiblePoint();

        // History of s (position differences) and y (gradient differences)
        std::vector<std::vector<double>> s_hist;
        std::vector<std::vector<double>> y_hist;

        auto grad = ComputeEIGradient(x);

        for (std::size_t iter = 0; iter < config_.lbfgs_max_iters; ++iter) {
            // Check convergence: max absolute gradient component
            double grad_inf = 0.0;
            for (double g : grad) {
                grad_inf = std::max(grad_inf, std::abs(g));
            }
            if (grad_inf < config_.lbfgs_grad_tol) break;

            // ── Two-loop recursion (L-BFGS) ──
            std::size_t hist_size = s_hist.size();
            std::vector<double> q = grad;  // Make a copy
            std::vector<double> rho(hist_size);
            std::vector<double> alpha_vals(hist_size);

            // First loop: iterate backward through history
            for (std::size_t i = hist_size; i-- > 0;) {
                double dot_sq = 0.0;
                for (std::size_t j = 0; j < dim; ++j) dot_sq += s_hist[i][j] * q[j];
                double dot_sy = 0.0;
                for (std::size_t j = 0; j < dim; ++j) dot_sy += s_hist[i][j] * y_hist[i][j];
                rho[i] = 1.0 / dot_sy;
                alpha_vals[i] = rho[i] * dot_sq;
                for (std::size_t j = 0; j < dim; ++j) q[j] -= alpha_vals[i] * y_hist[i][j];
            }

            // Scaling factor γ = s_last · y_last / y_last · y_last
            double gamma = 1.0;
            if (hist_size > 0) {
                double sy = 0.0;
                double yy = 0.0;
                for (std::size_t j = 0; j < dim; ++j) {
                    sy += s_hist.back()[j] * y_hist.back()[j];
                    yy += y_hist.back()[j] * y_hist.back()[j];
                }
                if (yy > 0.0) gamma = sy / yy;
            }
            std::vector<double> r(dim);
            for (std::size_t j = 0; j < dim; ++j) r[j] = gamma * q[j];

            // Second loop: iterate forward through history
            for (std::size_t i = 0; i < hist_size; ++i) {
                double dot_yr = 0.0;
                for (std::size_t j = 0; j < dim; ++j) dot_yr += y_hist[i][j] * r[j];
                double beta = rho[i] * dot_yr;
                for (std::size_t j = 0; j < dim; ++j) r[j] += s_hist[i][j] * (alpha_vals[i] - beta);
            }

            // Search direction (maximize, so use +r since we computed H·g)
            // For maximization: d = H * grad  (positive direction)
            std::vector<double> d = r;  // d = H * g

            // ── Backtracking line search with bound projection ──
            double step_size = 1.0;
            double armijo_c = 1e-4;
            double grad_dot_d = 0.0;
            for (std::size_t j = 0; j < dim; ++j) grad_dot_d += grad[j] * d[j];

            // EI is for maximization; ensure we move in an ascent direction
            // If dot product is negative, revert to gradient ascent step
            if (grad_dot_d <= 0.0) {
                d = grad;
                grad_dot_d = 0.0;
                for (std::size_t j = 0; j < dim; ++j) grad_dot_d += grad[j] * d[j];
            }

            double ei_x = ComputeEI(x);
            std::vector<double> x_new;

            while (step_size > 1e-10) {
                x_new.resize(dim);
                for (std::size_t j = 0; j < dim; ++j) x_new[j] = x[j] + step_size * d[j];
                space_.ClampToBounds(x_new);

                double ei_new = ComputeEI(x_new);

                // Armijo condition for maximization
                if (ei_new >= ei_x + armijo_c * step_size * grad_dot_d) {
                    break;
                }
                step_size *= 0.5;
            }

            // Compute gradient at new point
            auto grad_new = ComputeEIGradient(x_new);

            // Update history
            std::vector<double> s(dim);
            std::vector<double> y_vec(dim);
            double dot_sy = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                s[j] = x_new[j] - x[j];
                y_vec[j] = grad_new[j] - grad[j];
                dot_sy += s[j] * y_vec[j];
            }

            // Only update if curvature condition satisfied
            if (dot_sy > 1e-12) {
                if (s_hist.size() >= m) {
                    s_hist.erase(s_hist.begin());
                    y_hist.erase(y_hist.begin());
                }
                s_hist.push_back(std::move(s));
                y_hist.push_back(std::move(y_vec));
            }

            x = std::move(x_new);
            grad = std::move(grad_new);
        }

        double ei_final = ComputeEI(x);
        if (ei_final > best_ei) {
            best_ei = ei_final;
            best_x = std::move(x);
        }
    }

    return best_x;
}

// ── Utility ────────────────────────────────────────────────────────

auto BayesianOptimizer::RandomFeasiblePoint() const -> std::vector<double> {
    static thread_local std::mt19937 rng(std::random_device{}());

    std::vector<double> point;
    point.reserve(space_.Dimension());

    for (const auto& p : space_.Parameters()) {
        std::uniform_real_distribution<double> dist(p.min, p.max);
        double val = dist(rng);

        if (p.type == ParameterType::Discrete && p.step > 0.0) {
            double num_steps = std::round((val - p.min) / p.step);
            val = p.min + num_steps * p.step;
            val = std::clamp(val, p.min, p.max);
        }
        point.push_back(val);
    }
    return point;
}

// ── Main optimization loop ─────────────────────────────────────────

auto BayesianOptimizer::Optimize(ITuningObjective& objective,
                                  std::chrono::system_clock::time_point since)
    -> Result<OptimizationPoint> {
    if (space_.Dimension() == 0) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Tuning_SpaceEmpty,
                                             "TuningSpace has zero parameters — nothing to optimize"});
    }

    // Generate initial random points if needed
    while (observations_.size() < config_.initial_random_points) {
        auto point = RandomFeasiblePoint();
        auto cost_result = objective.Evaluate(point, since);
        if (!cost_result.has_value()) {
            return tl::make_unexpected(cost_result.error());
        }
        AddObservation({std::move(point), *cost_result, std::chrono::system_clock::now()});
    }

    std::size_t stagnation_count = 0;

    for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
        FitHyperparameters();
        cache_valid_ = false;  // Hyperparams may have changed

        auto best_x = LBFGSBMaximize();
        if (best_x.empty()) break;

        double ei_max = ComputeEI(best_x);

        if (ei_max < config_.ei_tol) {
            ++stagnation_count;
            if (stagnation_count >= config_.ei_stagnation_iters) break;
        } else {
            stagnation_count = 0;
        }

        // Evaluate objective at proposed point
        auto cost_result = objective.Evaluate(best_x, since);
        if (!cost_result.has_value()) {
            return tl::make_unexpected(cost_result.error());
        }

        AddObservation({std::move(best_x), *cost_result, std::chrono::system_clock::now()});
    }

    return BestPoint();
}

}  // namespace sai::tuning

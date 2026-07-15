// tuning_space.h — Batch T1: Bayesian Auto-Tuning search space definition
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai::tuning {

enum class ParameterType : std::uint8_t {
    Continuous,
    Discrete,
};

struct TuningParameter {
    std::string name;
    ParameterType type;
    double min;
    double max;
    double step{0.0};
};

struct ParameterConstraint {
    std::string lhs_name;
    double coeff_lhs{1.0};
    std::string rhs_name;
    double coeff_rhs{1.0};
    double bound{0.0};
};

class TuningSpace final {
public:
    auto AddParameter(TuningParameter param) -> TuningSpace&;
    auto AddConstraint(ParameterConstraint constraint) -> TuningSpace&;

    auto Parameters() const -> const std::vector<TuningParameter>&;
    auto Constraints() const -> const std::vector<ParameterConstraint>&;
    auto Dimension() const -> std::size_t;

    auto IsFeasible(const std::vector<double>& point) const -> bool;
    auto ClampToBounds(std::vector<double>& point) const -> void;

    static auto LoadFromYaml(const std::filesystem::path& path) -> Result<TuningSpace>;

private:
    auto FindIndex(const std::string& name) const -> std::size_t;

    std::vector<TuningParameter> parameters_;
    std::vector<ParameterConstraint> constraints_;
};

}  // namespace sai::tuning

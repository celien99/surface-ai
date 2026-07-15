// tuning_space.cpp — TuningSpace implementation + YAML parsing
#include <sai/tuning/tuning_space.h>

#include <algorithm>
#include <cmath>
#include <fstream>

#include <yaml-cpp/yaml.h>

namespace sai::tuning {

// ── Builder methods (fluent API) ─────────────────────────────────

auto TuningSpace::AddParameter(TuningParameter param) -> TuningSpace& {
    parameters_.push_back(std::move(param));
    return *this;
}

auto TuningSpace::AddConstraint(ParameterConstraint constraint) -> TuningSpace& {
    constraints_.push_back(std::move(constraint));
    return *this;
}

// ── Accessors ────────────────────────────────────────────────────

auto TuningSpace::Parameters() const -> const std::vector<TuningParameter>& {
    return parameters_;
}

auto TuningSpace::Constraints() const -> const std::vector<ParameterConstraint>& {
    return constraints_;
}

auto TuningSpace::Dimension() const -> std::size_t {
    return parameters_.size();
}

// ── Feasibility check ────────────────────────────────────────────

auto TuningSpace::IsFeasible(const std::vector<double>& point) const -> bool {
    if (point.size() != parameters_.size()) return false;

    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        const auto& p = parameters_[i];
        if (point[i] < p.min || point[i] > p.max) return false;

        if (p.type == ParameterType::Discrete && p.step > 0.0) {
            double num_steps = std::round((point[i] - p.min) / p.step);
            double snapped = p.min + num_steps * p.step;
            // Use a small epsilon to forgive floating-point rounding
            if (std::abs(point[i] - snapped) > 1e-9) return false;
        }
    }

    for (const auto& c : constraints_) {
        std::size_t li = FindIndex(c.lhs_name);
        std::size_t ri = FindIndex(c.rhs_name);
        if (li >= parameters_.size() || ri >= parameters_.size()) return false;
        double value = c.coeff_lhs * point[li] + c.coeff_rhs * point[ri];
        if (value > c.bound + 1e-9) return false;
    }

    return true;
}

// ── Clamp to valid region ────────────────────────────────────────

auto TuningSpace::ClampToBounds(std::vector<double>& point) const -> void {
    if (point.size() != parameters_.size()) return;

    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        const auto& p = parameters_[i];

        // 1. Clamp to [min, max]
        point[i] = std::clamp(point[i], p.min, p.max);

        // 2. For discrete parameters, snap to nearest valid step
        if (p.type == ParameterType::Discrete && p.step > 0.0) {
            double num_steps = std::round((point[i] - p.min) / p.step);
            point[i] = p.min + num_steps * p.step;
            // Defensive: clamp back in case rounding overflowed bound
            point[i] = std::clamp(point[i], p.min, p.max);
        }
    }
}

// ── YAML deserialization ─────────────────────────────────────────

auto TuningSpace::LoadFromYaml(const std::filesystem::path& path) -> Result<TuningSpace> {
    if (!std::filesystem::exists(path)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigFileNotFound,
            "tuning config file not found: " + path.string(),
            std::source_location::current(),
        });
    }

    TuningSpace space;
    try {
        YAML::Node root = YAML::LoadFile(path.string());
        auto tuning = root["tuning"];
        if (!tuning.IsDefined()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigParseError,
                "missing top-level 'tuning' key in: " + path.string(),
                std::source_location::current(),
            });
        }

        auto params = tuning["parameters"];
        if (!params.IsDefined() || !params.IsSequence()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigParseError,
                "missing or invalid 'tuning.parameters' list in: " + path.string(),
                std::source_location::current(),
            });
        }

        for (const auto& node : params) {
            TuningParameter p;
            p.name = node["name"].as<std::string>();
            p.min = node["min"].as<double>();
            p.max = node["max"].as<double>();

            std::string type_str = node["type"].as<std::string>();
            if (type_str == "continuous") {
                p.type = ParameterType::Continuous;
            } else if (type_str == "discrete") {
                p.type = ParameterType::Discrete;
            } else {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Infra_ConfigParseError,
                    "unknown parameter type '" + type_str + "' for parameter '" + p.name + "'",
                    std::source_location::current(),
                });
            }

            if (p.type == ParameterType::Discrete) {
                if (node["step"].IsDefined()) {
                    p.step = node["step"].as<double>();
                } else {
                    return tl::make_unexpected(ErrorInfo{
                        ErrorCode::Infra_ConfigParseError,
                        "discrete parameter '" + p.name + "' is missing 'step' field",
                        std::source_location::current(),
                    });
                }
            }

            space.AddParameter(std::move(p));
        }

        // Optional: parse constraints if present
        if (auto cons = tuning["constraints"]; cons.IsDefined() && cons.IsSequence()) {
            for (const auto& node : cons) {
                ParameterConstraint c;
                c.lhs_name = node["lhs"].as<std::string>();
                c.rhs_name = node["rhs"].as<std::string>();
                c.coeff_lhs = node["coeff_lhs"].as<double>(1.0);
                c.coeff_rhs = node["coeff_rhs"].as<double>(1.0);
                c.bound = node["bound"].as<double>();
                space.AddConstraint(std::move(c));
            }
        }
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigParseError,
            "YAML parse error in tuning config: " + std::string(e.what()),
            std::source_location::current(),
        });
    }

    if (space.Dimension() == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Tuning_SpaceEmpty,
            "tuning space has zero parameters after loading: " + path.string(),
            std::source_location::current(),
        });
    }

    return space;
}

// ── Private helpers ──────────────────────────────────────────────

auto TuningSpace::FindIndex(const std::string& name) const -> std::size_t {
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i].name == name) return i;
    }
    return parameters_.size();  // sentinel: not found
}

}  // namespace sai::tuning

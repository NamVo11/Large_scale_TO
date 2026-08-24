#pragma once

#include "myjson.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Analysis;

struct Function {
    std::string type = "constant";
    double amp = 0.0;
    double base_value = 0.0;

    Function() = default;

    explicit Function(const myjson::json& j) {
        type = myjson::lower_ascii(myjson::string_value(j, "type", myjson::string_value(j, "time_type", type)));
        amp = myjson::double_value(j, "amp", amp);
        base_value = myjson::double_value(j, "base_value", base_value);
    }

    virtual ~Function() = default;
    virtual double calculate(const Analysis& analysis, int istep, double scale) const = 0;
};

struct ConstantFunction : Function {
    ConstantFunction() = default;

    explicit ConstantFunction(const myjson::json& j)
        : Function(j) {}

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct LinearFunction : Function {
    LinearFunction() = default;

    explicit LinearFunction(const myjson::json& j)
        : Function(j) {}

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct SinFunction : Function {
    double nCycle = 0.0;
    double phase = 0.0;

    SinFunction() = default;

    explicit SinFunction(const myjson::json& j)
        : Function(j) {
        nCycle = myjson::double_value(j, "nCycle", nCycle);
        phase = myjson::double_value(j, "phase", phase);
    }

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct SumSinFunction : Function {
    double freq_factor = 1.0;
    double rise_time = 0.0;
    double phi_scale = 1.0;
    std::vector<double> freq;
    std::vector<double> scale;
    std::vector<double> phi;

    SumSinFunction() = default;

    explicit SumSinFunction(const myjson::json& j)
        : Function(j) {
        freq_factor = myjson::double_value(j, "freq_factor", freq_factor);
        rise_time = myjson::double_value(j, "rise_time", rise_time);
        phi_scale = myjson::double_value(j, "phi_scale", phi_scale);
        if (j.contains("freq")) j.at("freq").get_to(freq);
        if (j.contains("scale")) j.at("scale").get_to(scale);
        if (j.contains("phi")) j.at("phi").get_to(phi);
    }

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

inline std::shared_ptr<Function> InitializeFunction(const myjson::json& j) {
    const std::string type = myjson::lower_ascii(myjson::string_value(j, "type", myjson::string_value(j, "time_type", "constant")));
    if (type == "constant") return std::make_shared<ConstantFunction>(j);
    if (type == "linear") return std::make_shared<LinearFunction>(j);
    if (type == "sin") return std::make_shared<SinFunction>(j);
    if (type == "sum_sin" || type == "sum_sim" || type == "sum_sin1" || type == "sum_sim1" ||
        type == "sum_sin2" || type == "sum_sim2") {
        return std::make_shared<SumSinFunction>(j);
    }
    throw std::runtime_error("Unsupported BoundaryCondition Function.type: " + type);
}

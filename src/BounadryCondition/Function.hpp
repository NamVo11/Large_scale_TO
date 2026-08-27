#pragma once

#include "myjson.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Analysis;

struct Function {
    std::string type;
    double amp;
    double base_value = 0.0;

    Function() = delete;

    explicit Function(const myjson::json& j) {
        type = myjson::lower_ascii(j.at("type").get<std::string>());
        amp = j.at("amp").get<double>();
        base_value = myjson::double_value(j, "base_value", base_value);
    }

    virtual ~Function() = default;
    virtual double calculate(const Analysis& analysis, int istep, double scale) const = 0;
};

struct ConstantFunction : Function {
    explicit ConstantFunction(const myjson::json& j) : Function(j) {}

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct LinearFunction : Function {
    explicit LinearFunction(const myjson::json& j) : Function(j) {}

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct SinFunction : Function {
    double nCycle;
    double phase = 0.0;

    explicit SinFunction(const myjson::json& j) : Function(j) {
        nCycle = j.at("nCycle").get<double>();
        phase = myjson::double_value(j, "phase", phase);
    }

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

struct SumSinFunction : Function {
    double freq_factor = 1;
    double rise_time = 0;
    double phi_scale = 1;
    std::vector<double> freq;
    std::vector<double> scale;
    std::vector<double> phi;

    explicit SumSinFunction(const myjson::json& j) : Function(j) {
        freq_factor = myjson::double_value(j, "freq_factor", freq_factor);
        rise_time = myjson::double_value(j, "rise_time", rise_time);
        phi_scale = myjson::double_value(j, "phi_scale", phi_scale);
        j.at("freq").get_to(freq);
        j.at("scale").get_to(scale);
        j.at("phi").get_to(phi);
    }

    double calculate(const Analysis& analysis, int istep, double scale) const override;
};

inline std::shared_ptr<Function> InitializeFunction(const myjson::json& j) {

    const std::string type = myjson::lower_ascii(j.at("type").get<std::string>());
    if (type == "constant") return std::make_shared<ConstantFunction>(j);
    if (type == "linear") return std::make_shared<LinearFunction>(j);
    if (type == "sin") return std::make_shared<SinFunction>(j);
    if (type == "sum_sin") return std::make_shared<SumSinFunction>(j);

    throw std::runtime_error("Unsupported BoundaryCondition Function.type: " + type);
}

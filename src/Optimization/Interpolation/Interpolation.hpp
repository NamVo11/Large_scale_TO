#pragma once

#include "myjson.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Interpolation {
    int id = 0;
    std::string type;

    explicit Interpolation(const myjson::json& j) {
        id = j.at("id").get<int>();
        type = j.at("type").get<std::string>();
    }

    virtual ~Interpolation() = default;
    virtual double modulus_penalty() const { return 0.0; }
    virtual double mass_density_penalty() const { return 0.0; }

    virtual double calculate(const std::vector<double>& para,
                             double x,
                             bool is_design,
                             int val_id,
                             double p) const = 0;

    virtual double sensitivity(const std::vector<double>& para,
                               double x,
                               bool is_design,
                               int val_id,
                               double p) const = 0;
};

struct RAMP : Interpolation {
    double p_E_elastic = 0.0;
    double p_rho = 0.0;

    explicit RAMP(const myjson::json& j)
        : Interpolation(j) {
        p_E_elastic = j.at("p_E_elastic").get<double>();
        p_rho = j.at("p_rho").get<double>();
    }

    double modulus_penalty() const override { return p_E_elastic; }
    double mass_density_penalty() const override { return p_rho; }

    double calculate(const std::vector<double>& para,
                     double x,
                     bool is_design,
                     int val_id,
                     double p) const override;
    double sensitivity(const std::vector<double>& para,
                       double x,
                       bool is_design,
                       int val_id,
                       double p) const override;
};

struct SIMP : Interpolation {
    double penalty = 0.0;

    explicit SIMP(const myjson::json& j)
        : Interpolation(j) {
        penalty = j.at("penalty").get<double>();
    }

    double modulus_penalty() const override { return penalty; }
    double mass_density_penalty() const override { return penalty; }

    double calculate(const std::vector<double>& para,
                     double x,
                     bool is_design,
                     int val_id,
                     double p) const override;
    double sensitivity(const std::vector<double>& para,
                       double x,
                       bool is_design,
                       int val_id,
                       double p) const override;
};

struct DMO : Interpolation {
    explicit DMO(const myjson::json& j)
        : Interpolation(j) {}

    double calculate(const std::vector<double>& para,
                     double x,
                     bool is_design,
                     int val_id,
                     double p) const override;
    double sensitivity(const std::vector<double>& para,
                       double x,
                       bool is_design,
                       int val_id,
                       double p) const override;
};

inline std::shared_ptr<Interpolation> InitializeInterpolation(const myjson::json& j) {
    const std::string type = myjson::lower_ascii(j.at("type").get<std::string>());
    if (type == "ramp" || type == "ram") return std::make_shared<RAMP>(j);
    if (type == "simp") return std::make_shared<SIMP>(j);
    if (type == "dmo") return std::make_shared<DMO>(j);
    throw std::runtime_error("Unknown Interpolation.type: " + type);
}

inline std::vector<std::shared_ptr<Interpolation>> InitializeInterpolations(const myjson::json& j) {
    if (!j.is_array() || j.empty()) throw std::runtime_error("Interp must be a non-empty array");
    std::vector<std::shared_ptr<Interpolation>> interps;
    interps.reserve(j.size());
    for (const auto& item : j) interps.push_back(InitializeInterpolation(item));
    return interps;
}

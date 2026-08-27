#pragma once

#include "Analysis/Analysis.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Filtering {
    std::string type;
    double rmin = 3.0;
    double beta = 1.0;
    int beta_loop = 1000;
    double beta_max = 1.0;
    double T = 0.5;
    Eigen::MatrixXd xTilde;
    Eigen::MatrixXd xPhys;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<double>> weights;
    Eigen::SparseMatrix<double> H;
    std::vector<double> Hs;
    Eigen::VectorXd Hs_eigen;

    Filtering() = default;

    explicit Filtering(const myjson::json& j) {
        type = j.at("type").get<std::string>();
        rmin = j.at("rmin").get<double>();
        beta = myjson::double_value(j, "beta", beta);
        beta_loop = myjson::int_value(j, "beta_loop", beta_loop);
        beta_max = myjson::double_value(j, "beta_max", beta_max);
        T = myjson::double_value(j, "T", T);
    }

    virtual ~Filtering() = default;
    virtual void initialize(const FEM& fem);
    virtual void calculate(const Eigen::MatrixXd& x,
                           const std::vector<int>& x_label);
    virtual Eigen::MatrixXd sensitivity(const Eigen::MatrixXd& df_dx_phys) const;
};

struct DensityFiltering : Filtering {
    explicit DensityFiltering(const myjson::json& j) : Filtering(j) {}

    void initialize(const FEM& fem) override;
    void calculate(const Eigen::MatrixXd& x,
                   const std::vector<int>& x_label) override;
    Eigen::MatrixXd sensitivity(const Eigen::MatrixXd& df_dx_phys) const override;
};

struct SensitivityFiltering : Filtering {
    explicit SensitivityFiltering(const myjson::json& j) : Filtering(j) {}

    void initialize(const FEM& fem) override;
    void calculate(const Eigen::MatrixXd& x,
                   const std::vector<int>& x_label) override;
    Eigen::MatrixXd sensitivity(const Eigen::MatrixXd& df_dx_phys) const override;
};

inline std::unique_ptr<Filtering> InitializeFiltering(const myjson::json& j) {
    const std::string type = myjson::lower_ascii(j.at("type").get<std::string>());
    if (type == "densityfiltering" || type == "density_filtering") return std::make_unique<DensityFiltering>(j);
    if (type == "sensitivityfiltering" || type == "sensitivity_filtering") return std::make_unique<SensitivityFiltering>(j);
    throw std::runtime_error("Unknown Filtering.type: " + type);
}

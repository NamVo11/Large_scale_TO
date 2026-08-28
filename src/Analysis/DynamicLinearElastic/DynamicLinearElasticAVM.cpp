#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include <stdexcept>

void DynamicLinearElasticAVM::initialize_terms(const FEM& fem,
                                              Analysis& analysis,
                                              const Eigen::MatrixXd& xPhys) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    value = 0.0;
    for (auto& term : terms) {
        term->initialize(fem, dynamic_analysis, xPhys);
        value += term->scale * term->value;
    }
}

void DynamicLinearElasticAVM::calculate_terms(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    value = 0.0;
    for (auto& term : terms) {
        term->calculate(fem, dynamic_analysis, xPhys);
        value += term->scale * term->value;
    }
}

Eigen::VectorXd DynamicLinearElasticAVM::dfdu(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys,
                                             int istep) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdu(fem, dynamic_analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd DynamicLinearElasticAVM::dfdv(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys,
                                             int istep) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdv(fem, dynamic_analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd DynamicLinearElasticAVM::dfda(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys,
                                             int istep) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfda(fem, dynamic_analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd DynamicLinearElasticAVM::dfdFext(const FEM& fem,
                                                Analysis& analysis,
                                                const Eigen::MatrixXd& xPhys,
                                                int istep) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdFext(fem, dynamic_analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd DynamicLinearElasticAVM::dfdx(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys,
                                             int val_id) {
    auto& dynamic_analysis = static_cast<DynamicLinearElastic&>(analysis);
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdx(fem, dynamic_analysis, xPhys, val_id), term->scale);
    }
    return out;
}

Eigen::MatrixXd DynamicLinearElasticAVM::AVM(const FEM&,
                                            Analysis&,
                                            const Eigen::MatrixXd&,
                                            const std::vector<int>&,
                                            const Filtering&,
                                            int) {
    throw std::runtime_error("DynamicLinearElasticAVM::AVM backend implementation is not migrated yet");
}

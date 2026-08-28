#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"

#include <stdexcept>

void StaticLinearElasticAVM::initialize_terms(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys) {
    value = 0.0;
    for (auto& term : terms) {
        term->initialize(fem, analysis, xPhys);
        value += term->scale * term->value;
    }
}

void StaticLinearElasticAVM::calculate_terms(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& xPhys) {
    value = 0.0;
    for (auto& term : terms) {
        term->calculate(fem, analysis, xPhys);
        value += term->scale * term->value;
    }
}

Eigen::VectorXd StaticLinearElasticAVM::dfdu(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& xPhys,
                                            int istep) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdu(fem, analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd StaticLinearElasticAVM::dfdFext(const FEM& fem,
                                               Analysis& analysis,
                                               const Eigen::MatrixXd& xPhys,
                                               int istep) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdFext(fem, analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd StaticLinearElasticAVM::dfdx(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& xPhys,
                                            int val_id) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdx(fem, analysis, xPhys, val_id), term->scale);
    }
    return out;
}

Eigen::MatrixXd StaticLinearElasticAVM::AVM(const FEM&,
                                           Analysis&,
                                           const Eigen::MatrixXd&,
                                           const std::vector<int>&,
                                           const Filtering&,
                                           int) {
    throw std::runtime_error("StaticLinearElasticAVM::AVM requires a backend derived object");
}

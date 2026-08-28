#include "Optimization/Volume/VolumeFractionConstraint.hpp"

#include "Optimization/Filtering/Filtering.hpp"

#include <stdexcept>

void VolumeFractionConstraint::initialize_terms(const FEM& fem,
                                                Analysis& analysis,
                                                const Eigen::MatrixXd& xPhys) {
    value = 0.0;
    for (auto& term : terms) {
        term->initialize(fem, analysis, xPhys);
        value += term->scale * term->value;
    }
}

void VolumeFractionConstraint::calculate_terms(const FEM& fem,
                                               Analysis& analysis,
                                               const Eigen::MatrixXd& xPhys) {
    value = 0.0;
    for (auto& term : terms) {
        term->calculate(fem, analysis, xPhys);
        value += term->scale * term->value;
    }
}

Eigen::VectorXd VolumeFractionConstraint::dfdu(const FEM& fem,
                                               Analysis& analysis,
                                               const Eigen::MatrixXd& xPhys,
                                               int istep) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdu(fem, analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd VolumeFractionConstraint::dfdFext(const FEM& fem,
                                                  Analysis& analysis,
                                                  const Eigen::MatrixXd& xPhys,
                                                  int istep) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdFext(fem, analysis, xPhys, istep), term->scale);
    }
    return out;
}

Eigen::VectorXd VolumeFractionConstraint::dfdx(const FEM& fem,
                                               Analysis& analysis,
                                               const Eigen::MatrixXd& xPhys,
                                               int val_id) {
    Eigen::VectorXd out;
    for (auto& term : terms) {
        add_scaled(out, term->dfdx(fem, analysis, xPhys, val_id), term->scale);
    }
    return out;
}

Eigen::MatrixXd VolumeFractionConstraint::AVM(const FEM& fem,
                                              Analysis& analysis,
                                              const Eigen::MatrixXd& x,
                                              const std::vector<int>& x_label,
                                              const Filtering& filtering,
                                              int) {
    const int nvar = static_cast<int>(filtering.xPhys.cols());
    Eigen::MatrixXd dfdx_phys = Eigen::MatrixXd::Zero(filtering.xPhys.rows(), nvar);
    for (int val_id = 0; val_id < nvar; ++val_id) {
        const Eigen::VectorXd explicit_dfdx = dfdx(fem, analysis, filtering.xPhys, val_id);
        if (explicit_dfdx.size() == 0) continue;
        if (explicit_dfdx.size() != dfdx_phys.rows()) {
            throw std::runtime_error("VolumeFractionConstraint::AVM: dfdx size mismatch");
        }
        dfdx_phys.col(val_id).noalias() += explicit_dfdx;
    }
    (void)x;
    Eigen::MatrixXd out = filtering.sensitivity(dfdx_phys);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}

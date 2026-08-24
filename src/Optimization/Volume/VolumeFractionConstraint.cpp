#include "Optimization/Volume/VolumeFractionConstraint.hpp"

#include "Optimization/Filtering/Filtering.hpp"

#include <stdexcept>

void VolumeFractionConstraint::initialize_terms(const FEM& fem,
                                                Analysis& analysis,
                                                const Eigen::MatrixXd& xPhys) {
    for (auto& term : terms) {
        term->initialize(fem, analysis, xPhys);
    }
    accumulate_terms();
}

void VolumeFractionConstraint::calculate_terms(const FEM& fem,
                                               Analysis& analysis,
                                               const Eigen::MatrixXd& xPhys) {
    for (auto& term : terms) {
        term->calculate(fem, analysis, xPhys);
    }
    accumulate_terms();
}

Eigen::MatrixXd VolumeFractionConstraint::AVM(const FEM&,
                                              Analysis&,
                                              const Eigen::MatrixXd&,
                                              const std::vector<int>& x_label,
                                              const Filtering& filtering,
                                              const FilteringState& filtering_state,
                                              int) {
    Eigen::MatrixXd out = filtering.sensitivity(d_explicit_df_dx, filtering_state);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}

void VolumeFractionConstraint::accumulate_terms() {
    reset_accumulated_derivatives();
    for (const auto& term : terms) {
        if (term->initial_value == 0.0) {
            throw std::runtime_error("Volume functional has zero initial_value: " + term->function_name);
        }
        const double alpha = term->weight / term->initial_value;
        add_term_contribution(alpha,
                              term->value,
                              term->dfdu_all,
                              term->dfdv_all,
                              term->dfda_all,
                              term->dfdFext_all,
                              term->df_dx);
    }
}

#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"

#include <stdexcept>

void StaticLinearElasticAVM::initialize_terms(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys) {
    auto* static_analysis = dynamic_cast<StaticLinearElastic*>(&analysis);
    if (!static_analysis) throw std::runtime_error("StaticLinearElasticAVM requires StaticLinearElastic analysis");
    for (auto& term : terms) {
        term->initialize(fem, *static_analysis, xPhys);
    }
    accumulate_terms();
}

void StaticLinearElasticAVM::calculate_terms(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& xPhys) {
    auto* static_analysis = dynamic_cast<StaticLinearElastic*>(&analysis);
    if (!static_analysis) throw std::runtime_error("StaticLinearElasticAVM requires StaticLinearElastic analysis");
    for (auto& term : terms) {
        term->calculate(fem, *static_analysis, xPhys);
    }
    accumulate_terms();
}

Eigen::MatrixXd StaticLinearElasticAVM::AVM(const FEM&,
                                           Analysis&,
                                           const Eigen::MatrixXd&,
                                           const std::vector<int>&,
                                           const Filtering&,
                                           const FilteringState&,
                                           int) {
    throw std::runtime_error("StaticLinearElasticAVM::AVM requires a backend derived object");
}

void StaticLinearElasticAVM::accumulate_terms() {
    reset_accumulated_derivatives();
    for (const auto& term : terms) {
        if (term->initial_value == 0.0) {
            throw std::runtime_error("StaticLinearElastic functional has zero initial_value: " + term->function_name);
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

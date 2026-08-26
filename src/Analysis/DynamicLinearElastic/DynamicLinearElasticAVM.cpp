#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include <stdexcept>

void DynamicLinearElasticAVM::initialize_terms(const FEM& fem,
                                              Analysis& analysis,
                                              const Eigen::MatrixXd& xPhys) {
    auto* dynamic_analysis = dynamic_cast<DynamicLinearElastic*>(&analysis);
    if (!dynamic_analysis) throw std::runtime_error("DynamicLinearElasticAVM requires DynamicLinearElastic analysis");
    for (auto& term : terms) {
        term->initialize(fem, *dynamic_analysis, xPhys);
    }
    accumulate_terms();
}

void DynamicLinearElasticAVM::calculate_terms(const FEM& fem,
                                             Analysis& analysis,
                                             const Eigen::MatrixXd& xPhys) {
    auto* dynamic_analysis = dynamic_cast<DynamicLinearElastic*>(&analysis);
    if (!dynamic_analysis) throw std::runtime_error("DynamicLinearElasticAVM requires DynamicLinearElastic analysis");
    for (auto& term : terms) {
        term->calculate(fem, *dynamic_analysis, xPhys);
    }
    accumulate_terms();
}

Eigen::MatrixXd DynamicLinearElasticAVM::AVM(const FEM&,
                                            Analysis&,
                                            const Eigen::MatrixXd&,
                                            const std::vector<int>&,
                                            const Filtering&,
                                            int) {
    throw std::runtime_error("DynamicLinearElasticAVM::AVM backend implementation is not migrated yet");
}

void DynamicLinearElasticAVM::accumulate_terms() {
    reset_accumulated_derivatives();
    for (const auto& term : terms) {
        if (term->initial_value == 0.0) {
            throw std::runtime_error("DynamicLinearElastic functional has zero initial_value: " + term->function_name);
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

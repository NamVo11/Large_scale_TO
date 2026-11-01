#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"
#include "Analysis/Analysis.hpp"

#include <stdexcept>

StaticGeomericalNonlinearAVM::StaticGeomericalNonlinearAVM(
    const myjson::json& j,
    const std::vector<std::shared_ptr<Analysis>>& analyses,
    const myjson::json& analyses_json, const myjson::json& run_json)
    : GroupFunctionalBase(j, analyses, analyses_json, run_json) {
    if (!std::dynamic_pointer_cast<StaticGeomericalNonlinear>(analysis))
        throw std::runtime_error("Nonlinear AVM requires nonlinear static analysis");
    for (const auto& tj : j.at("terms"))
        terms.push_back(InitializeStaticGeomericalNonlinearFunctional(tj));
    if (terms.empty()) throw std::runtime_error("Nonlinear objective has no terms");
}
// ===========================================

void StaticGeomericalNonlinearAVM::initialize_terms(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys) {
    // Retained for the GroupFunctionalBase interface; run_nonlinear uses calculate_terms().
    for (auto& term : terms) {
        const double before = term->value;
        term->initialize(fem, A, xPhys); // initial_value = cumulative term.value
        value += term->scale * (term->value - before); // only this step
    }
}
// ===========================================

void StaticGeomericalNonlinearAVM::calculate_terms(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys) {
    for (auto& term : terms) term->calculate(fem, A, xPhys);
}
// ===========================================

double StaticGeomericalNonlinearAVM::step_value(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys) {
    double result = 0.0;
    for (auto& term : terms) result += term->scale * term->step_value(fem, A, xPhys);
    return result;
}
// ===========================================

Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdu(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys, int step) {
    Eigen::VectorXd out;
    for (auto& term : terms)
        add_scaled(out, term->dfdu(fem, A, xPhys, step), term->scale);
    return out;
}
// ===========================================

Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdFext(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys, int step) {
    Eigen::VectorXd out;
    for (auto& term : terms)
        add_scaled(out, term->dfdFext(fem, A, xPhys, step), term->scale);
    return out;
}
// ===========================================

Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdx(
    const FEM& fem, Analysis& A, const Eigen::MatrixXd& xPhys, int val_id) {
    Eigen::VectorXd out;
    for (auto& term : terms)
        add_scaled(out, term->dfdx(fem, A, xPhys, val_id), term->scale);
    return out;
}
// ===========================================

Eigen::MatrixXd StaticGeomericalNonlinearAVM::AVM(
    const FEM&, Analysis&, const Eigen::MatrixXd&, const std::vector<int>&,
    const Filtering&, int) {
    throw std::runtime_error("Nonlinear AVM requires Eigen backend derived class");
}
// ===========================================

#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"
#include "Analysis/Analysis.hpp"
#include <stdexcept>

StaticGeomericalNonlinearAVM::StaticGeomericalNonlinearAVM(
    const myjson::json& j,
    const std::vector<std::shared_ptr<Analysis>>& analyses,
    const myjson::json& analyses_json, const myjson::json& run_json)
    : GroupFunctionalBase(j, analyses, analyses_json, run_json) {
    if (!std::dynamic_pointer_cast<StaticGeomericalNonlinear>(analysis))
        throw std::runtime_error("Nonlinear AVM requires StaticGeomericalNonlinear analysis");
    for (const auto& tj : j.at("terms"))
        terms.push_back(InitializeStaticGeomericalNonlinearFunctional(tj));
    if (terms.empty()) throw std::runtime_error("Nonlinear functional group has no terms");
}

void StaticGeomericalNonlinearAVM::initialize_terms(const FEM& fem, Analysis& A,
                                                     const Eigen::MatrixXd& xPhys) {
    // Normalization needs the full trajectory: NEVER initialize scales here.
    calculate_terms(fem, A, xPhys);
}
void StaticGeomericalNonlinearAVM::calculate_terms(const FEM& fem, Analysis& A,
                                                   const Eigen::MatrixXd& xPhys) {
    value = 0.0;
    for (auto& term : terms) {
        term->calculate(fem, A, xPhys); // current step only
        value += term->scale * term->value;
    }
}
Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdu(const FEM& fem, Analysis& A,
                                                   const Eigen::MatrixXd& xPhys, int step) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(fem.ndof);
    for (auto& term : terms) v += term->scale * term->dfdu(fem, A, xPhys, step);
    return v;
}
Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdFext(const FEM& fem, Analysis& A,
                                                      const Eigen::MatrixXd& xPhys, int step) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(fem.ndof);
    for (auto& term : terms) v += term->scale * term->dfdFext(fem, A, xPhys, step);
    return v;
}
Eigen::VectorXd StaticGeomericalNonlinearAVM::dfdx(const FEM& fem, Analysis& A,
                                                   const Eigen::MatrixXd& xPhys, int col) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(fem.NElem);
    for (auto& term : terms) {
        const auto d = term->dfdx(fem, A, xPhys, col);
        if (d.size()) v += term->scale * d;
    }
    return v;
}

Eigen::MatrixXd StaticGeomericalNonlinearAVM::AVM(
    const FEM&, Analysis&, const Eigen::MatrixXd&, const std::vector<int>&,
    const Filtering&, int) {
    throw std::runtime_error("Nonlinear AVM requires Eigen backend derived class");
}

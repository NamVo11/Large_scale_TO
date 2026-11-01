#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearFunctional.hpp"
#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

StaticGeomericalNonlinear& StaticGeomericalNonlinearFunctionalBase::current_analysis(Analysis& analysis) {
    auto* A = dynamic_cast<StaticGeomericalNonlinear*>(&analysis);
    if (!A || A->current_step < 1 || A->U.cols() != 1 || A->F_ext.cols() != 1)
        throw std::runtime_error("Nonlinear functional needs converged current step");
    return *A;
}
// ===========================================

void StaticGeomericalNonlinearFunctionalBase::initialize(
    const FEM& fem, Analysis& analysis, const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    // value already accumulates across the load steps of iteration zero.
    initial_value = value;
}
// ===========================================

void StaticGeomericalNonlinearFunctionalBase::calculate(
    const FEM& fem, Analysis& analysis, const Eigen::MatrixXd& xPhys) {
    value += step_value(fem, analysis, xPhys);
}
// ===========================================

double GeometricalNonlinearCompliance::step_value(
    const FEM&, Analysis& analysis, const Eigen::MatrixXd&) {
    auto& A = current_analysis(analysis);
    return A.F_ext.col(0).dot(A.U.col(0));
}
// ===========================================

Eigen::VectorXd GeometricalNonlinearCompliance::dfdu(
    const FEM&, Analysis& analysis, const Eigen::MatrixXd&, int) {
    return current_analysis(analysis).F_ext.col(0);
}
// ===========================================

Eigen::VectorXd GeometricalNonlinearCompliance::dfdFext(
    const FEM&, Analysis& analysis, const Eigen::MatrixXd&, int) {
    return current_analysis(analysis).U.col(0);
}
// ===========================================

#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearFunctional.hpp"
#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

namespace {
StaticGeomericalNonlinear& current_analysis(Analysis& A) {
    auto* n = dynamic_cast<StaticGeomericalNonlinear*>(&A);
    if (!n || n->current_step < 1 || n->U.cols() != 2 || n->F_ext.cols() != 2)
        throw std::runtime_error("Nonlinear functional requires a converged current load step");
    return *n;
}
}

void StaticGeomericalNonlinearFunctionalBase::initialize(
    const FEM& fem,
    Analysis& analysis,
    const Eigen::MatrixXd& xPhys)
{
    calculate(fem, analysis, xPhys);
    initial_value = value;
}

void GeometricalNonlinearCompliance::calculate(
    const FEM& fem,
    Analysis& analysis,
    const Eigen::MatrixXd& xPhys)
{
    (void)fem;
    (void)xPhys;

    auto& A = dynamic_cast<StaticGeomericalNonlinear&>(analysis);

    value += A.F_ext.col(1).dot(A.U.col(1));
}
Eigen::VectorXd GeometricalNonlinearCompliance::dfdu(const FEM&, Analysis& A,
                                                     const Eigen::MatrixXd&, int /*istep*/) {
    auto& n = current_analysis(A);
    // d(F_ext^T u)/du, external force held fixed.
    return n.F_ext.col(1);
}

Eigen::VectorXd GeometricalNonlinearCompliance::dfdFext(const FEM&, Analysis& A,
                                                        const Eigen::MatrixXd&, int /*istep*/) {
    auto& n = current_analysis(A);
    return n.U.col(1);
}

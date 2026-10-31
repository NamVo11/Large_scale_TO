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

void GeometricalNonlinearCompliance::calculate(const FEM&, Analysis& A,
                                               const Eigen::MatrixXd&) {
    auto& n = current_analysis(A);
    // Work-type compliance for current step, as in existing static linear code.
    // Not strain energy: in geometrically nonlinear analysis these differ.
    value = n.F_ext.col(1).dot(n.U.col(1));
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

#include "Analysis/StaticLinearElastic/StaticLinearElasticFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

void StaticLinearElasticFunctionalBase::initialize(const FEM& fem,
                                                  Analysis& analysis,
                                                  const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    initial_value = value;
    if (initial_value == 0.0) {
        throw std::runtime_error("StaticLinearElastic functional has zero initial_value: " + function_name);
    }
    scale = weight / initial_value;
}

void Compliance::calculate(const FEM& fem,
                           Analysis& analysis,
                           const Eigen::MatrixXd& xPhys) {
    (void)fem;
    (void)xPhys;
    const int response_col = analysis.Nstep;

    const auto u = analysis.U.col(response_col);
    const auto f = analysis.F_ext.col(response_col);
    value = f.dot(u);
}

Eigen::VectorXd Compliance::dfdu(const FEM& fem,
                                 Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int istep) {
    (void)xPhys;
    (void)istep;
    const int response_col = analysis.Nstep;
    Eigen::VectorXd out = Eigen::VectorXd::Zero(fem.ndof);
    out = analysis.F_ext.col(response_col);
    for (int d : analysis.SPCdofs) {
        out(d) = 0.0;
    }
    return out;
}

Eigen::VectorXd Compliance::dfdFext(const FEM& fem,
                                    Analysis& analysis,
                                    const Eigen::MatrixXd& xPhys,
                                    int istep) {
    (void)xPhys;
    (void)istep;
    const int response_col = analysis.Nstep;
    Eigen::VectorXd out = Eigen::VectorXd::Zero(fem.ndof);
    out = analysis.U.col(response_col);
    for (int d : analysis.free) {
        out(d) = 0.0;
    }
    return out;
}

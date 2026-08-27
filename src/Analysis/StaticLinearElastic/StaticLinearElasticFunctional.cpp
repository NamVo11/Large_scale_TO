#include "Analysis/StaticLinearElastic/StaticLinearElasticFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

void StaticLinearElasticFunctionalBase::initialize(const FEM& fem,
                                                  StaticLinearElastic& analysis,
                                                  const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    if (initial_value == 0.0) initial_value = value;
}

void Compliance::calculate(const FEM& fem,
                           StaticLinearElastic& analysis,
                           const Eigen::MatrixXd& xPhys) {
    (void)xPhys;
    const int ndof = fem.ndof;
    const int ncols = analysis.Nstep + 1;
    const int response_col = analysis.Nstep;

    value = 0.0;
    dfdu_all = Eigen::MatrixXd::Zero(ndof, ncols);
    dfdv_all.resize(0, 0);
    dfda_all.resize(0, 0);
    dfdFext_all = Eigen::MatrixXd::Zero(ndof, ncols);
    df_dx.resize(0, 0);

    const auto u = analysis.U.col(response_col);
    const auto f = analysis.F_ext.col(response_col);
    value = f.dot(u);
    dfdu_all.col(response_col) = f;
    dfdFext_all.col(response_col) = u;
    for (int d : analysis.SPCdofs) {
        dfdu_all(d, response_col) = 0.0;
    }
    for (int d : analysis.free) {
        dfdFext_all(d, response_col) = 0.0;
    }
}

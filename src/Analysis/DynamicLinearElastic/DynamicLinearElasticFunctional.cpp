#include "Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

#include <cstdlib>

#include <iostream>

void DynamicLinearElasticFunctionalBase::initialize(const FEM& fem,
                                                   DynamicLinearElastic& analysis,
                                                   const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    if (initial_value == 0.0) initial_value = value;
    std::cout << "============================= " << std::endl;
    std::cout << "============================= " << std::endl;
    std::cout << "============================= " << initial_value << std::endl;
    std::exit(0);
}

void L2ComplianceElastic::calculate(const FEM& fem,
                                    DynamicLinearElastic& analysis,
                                    const Eigen::MatrixXd& xPhys) {
    (void)xPhys;
    const int ndof = fem.ndof;
    const int nstep = analysis.Nstep;

    value = 0.0;
    dfdu_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
    dfdv_all.resize(0, 0);
    dfda_all.resize(0, 0);
    dfdFext_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
    df_dx.resize(0, 0);

    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.results.U.col(k);
        const auto f = analysis.results.F_ext.col(k);
        const double s = f.dot(u);
        value += s * s;
        dfdu_all.col(k).noalias() = 2.0 * s * f;
        dfdFext_all.col(k).noalias() = 2.0 * s * u;
        for (int d : analysis.results.SPCdofs) {
            dfdu_all(d, k) = 0.0;
        }
        for (int d : analysis.results.free) {
            dfdFext_all(d, k) = 0.0;
        }
    }
}

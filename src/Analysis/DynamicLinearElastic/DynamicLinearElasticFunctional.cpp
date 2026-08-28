#include "Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

void DynamicLinearElasticFunctionalBase::initialize(const FEM& fem,
                                                   DynamicLinearElastic& analysis,
                                                   const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    initial_value = value;
    if (initial_value == 0.0) {
        throw std::runtime_error("DynamicLinearElastic functional has zero initial_value: " + function_name);
    }
    scale = weight / initial_value;
}

void L2ComplianceElastic::calculate(const FEM& fem,
                                    DynamicLinearElastic& analysis,
                                    const Eigen::MatrixXd& xPhys) {
    (void)fem;
    (void)xPhys;
    const int nstep = analysis.Nstep;

    value = 0.0;
    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.U.col(k);
        const auto f = analysis.F_ext.col(k);
        const double s = f.dot(u);
        value += s * s;
    }
}

Eigen::VectorXd L2ComplianceElastic::dfdu(const FEM& fem,
                                          DynamicLinearElastic& analysis,
                                          const Eigen::MatrixXd& xPhys,
                                          int istep) {
    (void)xPhys;
    if (istep < 0 || istep > analysis.Nstep) throw std::runtime_error("L2ComplianceElastic::dfdu: invalid step");
    Eigen::VectorXd out = Eigen::VectorXd::Zero(fem.ndof);
    const auto u = analysis.U.col(istep);
    const auto f = analysis.F_ext.col(istep);
    const double s = f.dot(u);
    out.noalias() = 2.0 * s * f;
    for (int d : analysis.SPCdofs) {
        out(d) = 0.0;
    }
    return out;
}

Eigen::VectorXd L2ComplianceElastic::dfdFext(const FEM& fem,
                                             DynamicLinearElastic& analysis,
                                             const Eigen::MatrixXd& xPhys,
                                             int istep) {
    (void)xPhys;
    if (istep < 0 || istep > analysis.Nstep) throw std::runtime_error("L2ComplianceElastic::dfdFext: invalid step");
    Eigen::VectorXd out = Eigen::VectorXd::Zero(fem.ndof);
    const auto u = analysis.U.col(istep);
    const auto f = analysis.F_ext.col(istep);
    const double s = f.dot(u);
    out.noalias() = 2.0 * s * u;
    for (int d : analysis.free) {
        out(d) = 0.0;
    }
    return out;
}

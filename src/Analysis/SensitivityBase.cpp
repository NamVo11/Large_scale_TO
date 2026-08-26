#include "Analysis/SensitivityBase.hpp"

#include "Analysis/Analysis.hpp"
#include "Output/OutputData.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace {

void add_scaled(Eigen::MatrixXd& dst, const Eigen::MatrixXd& src, double scale) {
    if (src.size() == 0) return;
    if (dst.size() == 0) {
        dst = scale * src;
        return;
    }
    dst.noalias() += scale * src;
}

} // namespace

void GroupFunctionalBase::debug_FDM(const FEM& fem,
                                    Analysis& analysis,
                                    const Eigen::MatrixXd& x,
                                    const std::vector<int>& x_label,
                                    const Filtering& filtering,
                                    const OutputData& output,
                                    double fd_epsilon,
                                    int max_checks,
                                    int Iter,
                                    const Eigen::MatrixXd& avm) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::ofstream out;
    if (rank == 0) {
        std::filesystem::create_directories(output.outDir_mat);
        const std::string path = output.outDir_mat + "/debug_sensitivity_fd.csv";
        static std::set<std::string> initialized_paths;
        const bool reset_file = Iter == 0 && initialized_paths.insert(path).second;
        out.open(path, reset_file ? std::ios::out : std::ios::app);
        if (!out) throw std::runtime_error("Cannot write debug_sensitivity_fd.csv");
        if (reset_file) {
            out << "Iter,group,elem,var,x,AVM,FD,abs_err,rel_err,value_plus,value_minus\n";
        }
    }

    const double current_value = value;
    int checked = 0;
    for (int e = 0; e < x.rows() && checked < max_checks; ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) continue;
        for (int v = 0; v < x.cols() && checked < max_checks; ++v) {
            const double x0 = x(e, v);
            const double h_plus = std::min(fd_epsilon, 1.0 - x0);
            const double h_minus = std::min(fd_epsilon, x0);
            if (h_plus <= 0.0 && h_minus <= 0.0) continue;

            double value_plus = current_value;
            double value_minus = current_value;
            if (h_plus > 0.0) {
                Eigen::MatrixXd xp = x;
                Filtering state_plus = filtering;
                xp(e, v) = x0 + h_plus;
                state_plus.calculate(xp, x_label);
                analysis.solver(fem, state_plus.xPhys, Iter);
                calculate_terms(fem, analysis, state_plus.xPhys);
                value_plus = value;
            }
            if (h_minus > 0.0) {
                Eigen::MatrixXd xm = x;
                Filtering state_minus = filtering;
                xm(e, v) = x0 - h_minus;
                state_minus.calculate(xm, x_label);
                analysis.solver(fem, state_minus.xPhys, Iter);
                calculate_terms(fem, analysis, state_minus.xPhys);
                value_minus = value;
            }

            double fd = 0.0;
            if (h_plus > 0.0 && h_minus > 0.0) fd = (value_plus - value_minus) / (h_plus + h_minus);
            else if (h_plus > 0.0) fd = (value_plus - current_value) / h_plus;
            else fd = (current_value - value_minus) / h_minus;

            if (rank == 0) {
                const double g = avm(e, v);
                const double abs_err = std::abs(g - fd);
                const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                out << Iter << "," << sens_function << "," << e << "," << v << "," << x0 << ","
                    << g << "," << fd << "," << abs_err << "," << rel_err << ","
                    << value_plus << "," << value_minus << "\n";
            }
            ++checked;
        }
    }

    analysis.solver(fem, filtering.xPhys, Iter);
    calculate_terms(fem, analysis, filtering.xPhys);
}

void GroupFunctionalBase::reset_accumulated_derivatives() {
    value = 0.0;
    d_explicit_dfdu_all.resize(0, 0);
    d_explicit_dfdv_all.resize(0, 0);
    d_explicit_dfda_all.resize(0, 0);
    d_explicit_dfdFext_all.resize(0, 0);
    d_explicit_df_dx.resize(0, 0);
}

void GroupFunctionalBase::add_term_contribution(double alpha,
                                                double term_value,
                                                const Eigen::MatrixXd& dfdu_all,
                                                const Eigen::MatrixXd& dfdv_all,
                                                const Eigen::MatrixXd& dfda_all,
                                                const Eigen::MatrixXd& dfdFext_all,
                                                const Eigen::MatrixXd& df_dx) {
    value += alpha * term_value;
    add_scaled(d_explicit_dfdu_all, dfdu_all, alpha);
    add_scaled(d_explicit_dfdv_all, dfdv_all, alpha);
    add_scaled(d_explicit_dfda_all, dfda_all, alpha);
    add_scaled(d_explicit_dfdFext_all, dfdFext_all, alpha);
    add_scaled(d_explicit_df_dx, df_dx, alpha);
}

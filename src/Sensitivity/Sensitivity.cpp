#include "Sensitivity/Sensitivity.hpp"

#include "Analysis/Analysis.hpp"
#include "MatlabLikeData.hpp"
#include "Output/OutputData.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

void zero_labeled_rows(Eigen::MatrixXd& out, const std::vector<int>& x_label) {
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
}

double objective_group_value_for_debug(const FEM& fem,
                                       const Analysis& analysis,
                                       const Eigen::MatrixXd& xPhys,
                                       const ObjectiveGroup& object) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return 0.0;

    double value = 0.0;
    for (const auto& source_term : object.terms) {
        if (lower(source_term.function_name) != "l2_compliance_elastic") {
            throw std::runtime_error("Unknown objective term");
        }

        ObjectiveTerm term = source_term;
        const int ndof = fem.ndof;
        const int nstep = analysis.Nstep;
        term.value = 0.0;
        term.dfdu_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
        term.dfdv_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
        term.dfda_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
        term.dfdFext_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
        term.df_dx = Eigen::MatrixXd::Zero(fem.NElem, static_cast<int>(xPhys.cols()));
        for (int k = 0; k <= nstep; ++k) {
            const auto u = analysis.results.U.col(k);
            const auto f = analysis.results.F_ext.col(k);
            const double s = f.dot(u);
            term.value += s * s;
        }

        const double value_initial = source_term.value_initial != 0.0 ? source_term.value_initial : term.value;
        value += term.weight * term.value / value_initial;
    }
    return value;
}

} // namespace

Eigen::MatrixXd SensitivityBase::constraint_sensitivity(const FEM&,
                                                        Analysis&,
                                                        const Eigen::MatrixXd&,
                                                        const std::vector<int>& x_label,
                                                        Constraint& object,
                                                        const Filtering& filtering,
                                                        const FilteringState& filtering_state,
                                                        int) {  
    zero_labeled_rows(object.d_explicit_df_dx, x_label);                                                   
    Eigen::MatrixXd out = filtering_sensitivity(object.d_explicit_df_dx, filtering, filtering_state);
    zero_labeled_rows(out, x_label);
    return out;
}

Eigen::MatrixXd SensitivityBase::debug_objective_sensitivity(const FEM& fem,
                                                             Analysis& analysis,
                                                             const Eigen::MatrixXd& x,
                                                             const std::vector<int>& x_label,
                                                             ObjectiveGroup& object,
                                                             const Filtering& filtering,
                                                             const FilteringState& filtering_state,
                                                             const OutputData& output,
                                                             double fd_epsilon,
                                                             int max_checks,
                                                             int Iter) {
    Eigen::MatrixXd avm = objective_sensitivity(fem, analysis, x, x_label, object, filtering, filtering_state, Iter);

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::ofstream out;
    if (rank == 0) {
        std::filesystem::create_directories(output.outDir_mat);
        out.open(output.outDir_mat + "/debug_sensitivity_fd.csv", Iter == 0 ? std::ios::out : std::ios::app);
        if (!out) throw std::runtime_error("Cannot write debug_sensitivity_fd.csv");
        if (Iter == 0) {
            out << "Iter,elem,var,x,AVM,FD,abs_err,rel_err,obj_plus,obj_minus\n";
        }
    }

    int checked = 0;
    for (int e = 0; e < x.rows() && checked < max_checks; ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) continue;
        for (int v = 0; v < x.cols() && checked < max_checks; ++v) {
            const double x0 = x(e, v);
            const double h_plus = std::min(fd_epsilon, 1.0 - x0);
            const double h_minus = std::min(fd_epsilon, x0);
            if (h_plus <= 0.0 && h_minus <= 0.0) continue;

            double obj_plus = object.value;
            double obj_minus = object.value;
            if (h_plus > 0.0) {
                Eigen::MatrixXd xp = x;
                FilteringState state_plus = filtering_state;
                xp(e, v) = x0 + h_plus;
                filtering_calculation(xp, x_label, filtering, state_plus);
                analysis.solver(fem, state_plus.xPhys, Iter);
                obj_plus = objective_group_value_for_debug(fem, analysis, state_plus.xPhys, object);
            }
            if (h_minus > 0.0) {
                Eigen::MatrixXd xm = x;
                FilteringState state_minus = filtering_state;
                xm(e, v) = x0 - h_minus;
                filtering_calculation(xm, x_label, filtering, state_minus);
                analysis.solver(fem, state_minus.xPhys, Iter);
                obj_minus = objective_group_value_for_debug(fem, analysis, state_minus.xPhys, object);
            }

            double fd = 0.0;
            if (h_plus > 0.0 && h_minus > 0.0) fd = (obj_plus - obj_minus) / (h_plus + h_minus);
            else if (h_plus > 0.0) fd = (obj_plus - object.value) / h_plus;
            else fd = (object.value - obj_minus) / h_minus;

            if (rank == 0) {
                const double g = avm(e, v);
                const double abs_err = std::abs(g - fd);
                const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                out << Iter << "," << e << "," << v << "," << x0 << ","
                    << g << "," << fd << "," << abs_err << "," << rel_err << ","
                    << obj_plus << "," << obj_minus << "\n";
            }
            ++checked;
        }
    }

    analysis.solver(fem, filtering_state.xPhys, Iter);
    return avm;
}

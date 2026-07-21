#include "Eigen_backend/ROM/Utils/rom_eigen_utils.hpp"

#include <algorithm>
#include <cctype>

namespace eigen_backend {

std::string lower_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

double eigen_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                               const Eigen::VectorXd& rhs,
                               const Eigen::VectorXd& solution) {
    const double denom = std::max(1e-30, rhs.norm());
    return (lhs * solution - rhs).norm() / denom;
}

double eigen_relative_solution_error(const Eigen::VectorXd& fom_solution,
                                     const Eigen::VectorXd& rom_solution) {
    if (fom_solution.size() != rom_solution.size()) return 0.0;
    return (fom_solution - rom_solution).norm() / std::max(1e-30, fom_solution.norm());
}

double eigen_basis_error(const Eigen::MatrixXd& basis,
                         const Eigen::VectorXd& solution) {
    if (basis.cols() == 0 || solution.size() == 0) return 1.0;
    const Eigen::VectorXd projection = basis * (basis.transpose() * solution);
    return (solution - projection).norm() / std::max(1e-30, solution.norm());
}

} // namespace eigen_backend

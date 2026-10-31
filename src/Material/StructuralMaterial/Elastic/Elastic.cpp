#include "Material/StructuralMaterial/Elastic/Elastic.hpp"

#include <stdexcept>

void Elastic::cal_constitutive_law(const Eigen::MatrixXd& B,
                                  const Eigen::VectorXd& u_e,
                                  double young_modulus,
                                  double poisson_ratio) {
    set_isotropic_tangent(young_modulus, poisson_ratio);
    if (B.rows()!=C_matrix_.rows() || B.cols()!=u_e.size())
        throw std::invalid_argument("Elastic B/displacement size mismatch");

    // Linear assembly calls this method before solving, with u_e = 0, and
    // needs only C_matrix_ to form K_e = integral(B^T C B). Computing and
    // storing strain/stress here would misleadingly record the zero trial
    // state as the solved material-point response.
    // When post-processing or stress sensitivity is implemented, update the
    // material point using the solved u_e in a separate recovery step:
    // strain_ = B * u_e;
    // stress_ = C_matrix_ * strain_;
}

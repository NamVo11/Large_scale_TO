#include "FEM/Shape/Shape.hpp"

#include <stdexcept>

void Shape::update_at_gauss_point(std::size_t ip, const Eigen::MatrixXd& X) {
    const auto& rule = integration_rule();
    if (ip >= rule.size()) throw std::out_of_range("Gauss point index");
    if (X.rows() != numnode() || X.cols() != num_dimension())
        throw std::invalid_argument("Shape: reference coordinates have wrong dimensions");
    create_shape_function(rule[ip].xi, N_, dNdXi_);
    const Eigen::MatrixXd J = X.transpose() * dNdXi_;
    detJ0_ = J.determinant();
    if (!(detJ0_ > 0.0)) throw std::runtime_error("Shape: non-positive reference Jacobian");
    dNdX_ = dNdXi_ * J.inverse();
    factor_ = rule[ip].weight * detJ0_;
}

std::size_t Shape::num_integral_point() const { return integration_rule().size(); }
const Eigen::VectorXd& Shape::N() const { return N_; }
const Eigen::MatrixXd& Shape::dNdX() const { return dNdX_; }
double Shape::detJ0() const { return detJ0_; }
double Shape::factor() const { return factor_; }

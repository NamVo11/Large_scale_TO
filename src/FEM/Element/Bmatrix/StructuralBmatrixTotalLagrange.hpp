#pragma once

#include "FEM/Element/Bmatrix/Bmatrix.hpp"

// Total Lagrange: B_L(u_e), B_NL va Sbar dung cho material/geometric tangents.
class StructuralBmatrixTotalLagrange : public Bmatrix {
public:
    explicit StructuralBmatrixTotalLagrange(int dimension) : dim_(dimension) {}
    ~StructuralBmatrixTotalLagrange() override = default;
    Eigen::MatrixXd create(const Shape& shape, const Eigen::Matrix3d& F) const;
    Eigen::MatrixXd create_NL(const Shape& shape) const;
    Eigen::MatrixXd create_Sbar(const Eigen::VectorXd& stress) const;

protected:
    const int dim_;
};

class StructuralBmatrixTotalLagrange2D final : public StructuralBmatrixTotalLagrange {
public:
    StructuralBmatrixTotalLagrange2D() : StructuralBmatrixTotalLagrange(2) {}
};

class StructuralBmatrixTotalLagrange3D final : public StructuralBmatrixTotalLagrange {
public:
    StructuralBmatrixTotalLagrange3D() : StructuralBmatrixTotalLagrange(3) {}
};

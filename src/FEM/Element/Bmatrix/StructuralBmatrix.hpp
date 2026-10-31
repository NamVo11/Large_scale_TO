#pragma once

#include "FEM/Element/Bmatrix/Bmatrix.hpp"

// Small-strain B matrix; the dimension is fixed by the concrete formulation.
class StructuralBmatrix : public Bmatrix {
public:
    explicit StructuralBmatrix(int dimension) : dim_(dimension) {}
    ~StructuralBmatrix() override = default;
    Eigen::MatrixXd create(const Shape& shape,
                           const Eigen::VectorXd& u_e) const;

protected:
    const int dim_;
};

class StructuralBmatrix2D final : public StructuralBmatrix {
public:
    StructuralBmatrix2D() : StructuralBmatrix(2) {}
};

class StructuralBmatrix3D final : public StructuralBmatrix {
public:
    StructuralBmatrix3D() : StructuralBmatrix(3) {}
};

#include "FEM/Element/Bmatrix/StructuralBmatrix.hpp"

#include <stdexcept>

Eigen::MatrixXd StructuralBmatrix::create(const Shape& shape,
                                          const Eigen::VectorXd& u_e) const {
    if (shape.num_dimension()!=dim_ || u_e.size()!=dim_*shape.numnode())
        throw std::invalid_argument("Small-strain Bmatrix: shape/displacement dimension mismatch");

    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(dim_==2 ? 3 : 6,u_e.size());
    const auto& g=shape.dNdX();
    for (int a=0;a<shape.numnode();++a) {
        if (dim_==2) {
            B(0,2*a)=g(a,0);
            B(1,2*a+1)=g(a,1);
            B(2,2*a)=g(a,1);
            B(2,2*a+1)=g(a,0);
        } else {
            const int c=3*a;
            B(0,c)=g(a,0);
            B(1,c+1)=g(a,1);
            B(2,c+2)=g(a,2);
            B(3,c)=g(a,1);
            B(3,c+1)=g(a,0);
            B(4,c+1)=g(a,2);
            B(4,c+2)=g(a,1);
            B(5,c)=g(a,2);
            B(5,c+2)=g(a,0);
        }
    }
    return B;
}

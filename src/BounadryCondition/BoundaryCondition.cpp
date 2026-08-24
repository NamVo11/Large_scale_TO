#include "BounadryCondition/BoundaryCondition.hpp"

#include <stdexcept>

double BoundaryCondition::value(const Analysis& analysis, int istep, double scale) const {
    if (!function) throw std::runtime_error("BoundaryCondition has no Function object");
    if (!(scale > 0.0)) throw std::runtime_error("BoundaryCondition::value requires scale > 0");
    return function->calculate(analysis, istep, scale);
}

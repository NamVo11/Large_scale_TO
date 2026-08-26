#include "Optimization/Interpolation/Interpolation.hpp"

#include <algorithm>

namespace {

double fixed_value(const std::vector<double>& para, int val_id) {
    if (para.empty()) throw std::runtime_error("Interpolation needs at least one material parameter");
    const int idx = std::clamp(val_id, 0, static_cast<int>(para.size()) - 1);
    return para[static_cast<std::size_t>(idx)];
}

} // namespace

double RAMP::calculate(const std::vector<double>& para,
                       double x,
                       bool is_design,
                       int val_id,
                       double p) const {
    if (!is_design) return fixed_value(para, val_id);
    if (para.size() < 2) throw std::runtime_error("RAMP needs at least two parameters");
    const double denom = 1.0 + p * (1.0 - x);
    const double g = x / denom;
    return para[0] + (para[1] - para[0]) * g;
}

double RAMP::sensitivity(const std::vector<double>& para,
                         double x,
                         bool is_design,
                         int,
                         double p) const {
    if (!is_design) return 0.0;
    if (para.size() < 2) throw std::runtime_error("RAMP sensitivity needs at least two parameters");
    const double denom = 1.0 + p * (1.0 - x);
    return (para[1] - para[0]) * (1.0 + p) / (denom * denom);
}

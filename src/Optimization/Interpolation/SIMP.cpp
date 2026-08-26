#include "Optimization/Interpolation/Interpolation.hpp"

#include <algorithm>
#include <cmath>

namespace {

double fixed_value(const std::vector<double>& para, int val_id) {
    if (para.empty()) throw std::runtime_error("SIMP needs at least one material parameter");
    const int idx = std::clamp(val_id, 0, static_cast<int>(para.size()) - 1);
    return para[static_cast<std::size_t>(idx)];
}

} // namespace

double SIMP::calculate(const std::vector<double>& para,
                       double x,
                       bool is_design,
                       int val_id,
                       double p) const {
    if (!is_design) return fixed_value(para, val_id);
    if (para.size() < 2) throw std::runtime_error("SIMP design interpolation needs at least two parameters");
    return para[0] + (para[1] - para[0]) * std::pow(x, p);
}

double SIMP::sensitivity(const std::vector<double>& para,
                         double x,
                         bool is_design,
                         int,
                         double p) const {
    if (!is_design) return 0.0;
    if (para.size() < 2) throw std::runtime_error("SIMP sensitivity needs at least two parameters");
    return (para[1] - para[0]) * p * std::pow(std::max(0.0, x), p - 1.0);
}

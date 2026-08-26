#include "Optimization/Interpolation/Interpolation.hpp"

#include <algorithm>

double DMO::calculate(const std::vector<double>& para,
                      double,
                      bool,
                      int val_id,
                      double) const {
    if (para.empty()) throw std::runtime_error("DMO needs at least one material parameter");
    const int idx = std::clamp(val_id, 0, static_cast<int>(para.size()) - 1);
    return para[static_cast<std::size_t>(idx)];
}

double DMO::sensitivity(const std::vector<double>&,
                        double,
                        bool,
                        int,
                        double) const {
    return 0.0;
}

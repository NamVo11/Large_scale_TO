#include "BounadryCondition/Function.hpp"

#include "Analysis/Analysis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

const DynamicLinearElastic& dynamic_analysis(const Analysis& analysis, const std::string& function_name) {
    const auto* dynamic = dynamic_cast<const DynamicLinearElastic*>(&analysis);
    if (!dynamic) throw std::runtime_error("BC Function.type=" + function_name + " requires DynamicLinearElastic analysis");
    return *dynamic;
}

void validate_sum_sin_vectors(const SumSinFunction& function) {
    if (function.freq.empty()) throw std::runtime_error("SumSinFunction requires non-empty freq");
    if (function.scale.size() != function.freq.size()) throw std::runtime_error("SumSinFunction scale size mismatch");
    if (function.phi.size() != function.freq.size()) throw std::runtime_error("SumSinFunction phi size mismatch");
}

} // namespace

double ConstantFunction::calculate(const Analysis&, int, double scale) const {
    return amp / scale;
}

double LinearFunction::calculate(const Analysis& analysis, int istep, double scale) const {
    return (amp / scale) * static_cast<double>(istep) / static_cast<double>(analysis.Nstep);
}

double SinFunction::calculate(const Analysis& analysis, int istep, double scale) const {
    const DynamicLinearElastic& dynamic = dynamic_analysis(analysis, type);
    const double t = istep * dynamic.dt;
    return (base_value + amp * std::sin(2.0 * pi * nCycle * t / dynamic.Tmax + phase)) / scale;
}

double SumSinFunction::calculate(const Analysis& analysis, int istep, double scale) const {
    const DynamicLinearElastic& dynamic = dynamic_analysis(analysis, type);
    validate_sum_sin_vectors(*this);
    const double t = istep * dynamic.dt;
    double envelope = 1.0;
    if (rise_time > 0.0) envelope = std::min(t / rise_time, 1.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < freq.size(); ++i) {
        sum += this->scale[i] * std::sin(2.0 * pi * freq_factor * freq[i] * t + phi_scale * phi[i]);
    }
    return (base_value + amp * envelope * sum) / scale;
}

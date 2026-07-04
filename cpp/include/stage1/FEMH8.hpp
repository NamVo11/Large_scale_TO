#pragma once

#include "stage1/CaseData.hpp"

#include <array>

struct ElementMatrices {
    std::array<double, 24 * 24> ke{};
    std::array<double, 24 * 24> me{};
};

ElementMatrices computeH8ElementMatrices(const CaseData& data, int elemId);
double effectiveE(const MaterialData& mat, double x);
double effectiveRho(const MaterialData& mat, double x);

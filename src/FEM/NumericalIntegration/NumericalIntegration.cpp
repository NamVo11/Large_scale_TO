#include "FEM/NumericalIntegration/NumericalIntegration.hpp"

#include <cmath>
#include <stdexcept>

const std::vector<IntegrationPoint>& NumericalIntegration::quad4(int order) {
    if (order != 2) throw std::invalid_argument("Quad4 supports 2x2 integration");
    const double g = 1.0 / std::sqrt(3.0);
    static const std::vector<IntegrationPoint> rule =
        {{{-g,-g,0},1}, {{g,-g,0},1}, {{g,g,0},1}, {{-g,g,0},1}};
    return rule;
}

const std::vector<IntegrationPoint>& NumericalIntegration::hexa8(int order) {
    if (order != 2) throw std::invalid_argument("Hexa8 supports 2x2x2 integration");
    const double g = 1.0 / std::sqrt(3.0);
    static const std::vector<IntegrationPoint> rule = [g] {
        std::vector<IntegrationPoint> points;
        points.reserve(8);
        for (double z : {-g, g}) for (double y : {-g, g}) for (double x : {-g, g})
            points.push_back({{x,y,z},1.0});
        return points;
    }();
    return rule;
}

const std::vector<IntegrationPoint>& NumericalIntegration::tria3() {
    static const std::vector<IntegrationPoint> rule = {{{1.0/3.0,1.0/3.0,0},0.5}};
    return rule;
}

const std::vector<IntegrationPoint>& NumericalIntegration::tetra4() {
    static const std::vector<IntegrationPoint> rule = {{{0.25,0.25,0.25},1.0/6.0}};
    return rule;
}

const std::vector<IntegrationPoint>& NumericalIntegration::penta6() {
    const double g = 1.0 / std::sqrt(3.0);
    static const std::vector<IntegrationPoint> rule =
        {{{1.0/3.0,1.0/3.0,-g},0.5}, {{1.0/3.0,1.0/3.0,g},0.5}};
    return rule;
}

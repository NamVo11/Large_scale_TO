#include "Analysis/Analysis.hpp"

#include "FEM/Element/ElementBase.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

std::vector<int> ids_by_id(const std::vector<std::shared_ptr<BoundaryCondition>>& all, const std::vector<int>& ids) {
    std::vector<int> out;
    for (int id : ids) {
        auto it = std::find_if(all.begin(), all.end(), [id](const std::shared_ptr<BoundaryCondition>& b) {
            return b && b->id == id;
        });
        if (it == all.end()) throw std::runtime_error("Boundary condition id not found");
        out.push_back(static_cast<int>(std::distance(all.begin(), it)));
    }
    return out;
}

} // namespace

void Analysis::solver(const FEM&, const Eigen::MatrixXd&, int) {
    throw std::runtime_error("Analysis::solver is not implemented for base Analysis");
}

void Analysis::init_interp(const std::vector<std::shared_ptr<Interpolation>>& interps) {
    for (const auto& candidate : interps) {
        if (candidate && candidate->id == interp_id) {
            interp = candidate;
            return;
        }
    }
    throw std::runtime_error("Analysis interpolation id not found: " + std::to_string(interp_id));
}

void Analysis::attach_bcs_by_id(const std::vector<std::shared_ptr<BoundaryCondition>>& Dirich_all,
                                const std::vector<std::shared_ptr<BoundaryCondition>>& Neum_all) {
    Dirich.clear();
    Neum.clear();
    for (int idx : ids_by_id(Dirich_all, Dirich_id)) {
        Dirich.push_back(Dirich_all[static_cast<std::size_t>(idx)]);
    }
    for (int idx : ids_by_id(Neum_all, Neum_id)) {
        Neum.push_back(Neum_all[static_cast<std::size_t>(idx)]);
    }
}

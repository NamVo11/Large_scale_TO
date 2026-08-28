#include "Analysis/Analysis.hpp"

#include "FEM/Element/ElementBase.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

void push_bc_by_id(std::vector<std::shared_ptr<const BoundaryCondition>>& target,
                   const std::vector<std::shared_ptr<BoundaryCondition>>& source,
                   int bc_id,
                   const char* kind) {
    auto it = std::find_if(source.begin(), source.end(), [bc_id](const auto& bc) {
        return bc && bc->id == bc_id;
    });
    if (it == source.end()) {
        throw std::runtime_error(std::string(kind) + " boundary condition id not found: " + std::to_string(bc_id));
    }
    target.push_back(*it);
}

} // namespace

void Analysis::solver(const FEM&, const Eigen::MatrixXd&, int) {
    throw std::runtime_error("Analysis::solver is not implemented for base Analysis");
}

void Analysis::init_interp(const std::vector<std::shared_ptr<Interpolation>>& interps, int interp_id) {
    for (const auto& candidate : interps) {
        if (candidate && candidate->id == interp_id) {
            interp = candidate;
            return;
        }
    }
    throw std::runtime_error("Analysis interpolation id not found: " + std::to_string(interp_id));
}

void Analysis::attach_bcs_by_id(const myjson::json& dirich_ids,
                                const myjson::json& neum_ids,
                                const std::vector<std::shared_ptr<BoundaryCondition>>& dirich_all,
                                const std::vector<std::shared_ptr<BoundaryCondition>>& neum_all) {
    Dirich.clear();
    Neum.clear();
    for (const auto& bc_id_json : dirich_ids) {
        push_bc_by_id(Dirich, dirich_all, bc_id_json.get<int>(), "Dirichlet");
    }
    for (const auto& bc_id_json : neum_ids) {
        push_bc_by_id(Neum, neum_all, bc_id_json.get<int>(), "Neumann");
    }
}

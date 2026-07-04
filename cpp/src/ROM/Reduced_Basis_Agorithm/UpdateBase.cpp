#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"

namespace stage2 {

UpdateBase::UpdateBase(int max_rank_in) : max_rank(max_rank_in) {}

UpdateBase::~UpdateBase() {
    reset();
}

const std::vector<Vec>& UpdateBase::getBasis(const ROMContext&) const {
    return V;
}

void UpdateBase::setBasis(const std::vector<Vec>& V_in, const std::vector<double>& s_in) {
    clear_vectors(V);
    for (Vec v : V_in) V.push_back(duplicate_vec(v));
    s = s_in;
}

void UpdateBase::addSnapshot(Vec solution, const ROMContext&) {
    snapshots.push_back(duplicate_vec(solution));
}

void UpdateBase::clearSnapshots() {
    clear_vectors(snapshots);
}

void UpdateBase::reset() {
    clear_vectors(V);
    clear_vectors(snapshots);
    s.clear();
    W.clear();
}

Vec UpdateBase::duplicate_vec(Vec source) {
    Vec out = nullptr;
    VecDuplicate(source, &out);
    VecCopy(source, out);
    return out;
}

void UpdateBase::clear_vectors(std::vector<Vec>& vecs) {
    for (Vec& v : vecs) {
        if (v) VecDestroy(&v);
    }
    vecs.clear();
}

} // namespace stage2

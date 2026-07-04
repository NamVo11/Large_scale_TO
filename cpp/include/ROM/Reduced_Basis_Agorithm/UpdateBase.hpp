#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#include <petscvec.h>

#include <string>
#include <vector>

namespace stage2 {

class UpdateBase {
public:
    std::string name = "UpdateBase";
    int max_rank = 50;
    std::vector<Vec> V;
    std::vector<Vec> snapshots;
    std::vector<double> s;
    std::vector<std::vector<double>> W;

    explicit UpdateBase(int max_rank_in = 50);
    virtual ~UpdateBase();

    const std::vector<Vec>& getBasis(const ROMContext& context) const;
    void setBasis(const std::vector<Vec>& V_in, const std::vector<double>& s_in = {});
    void addSnapshot(Vec solution, const ROMContext& context);
    void clearSnapshots();
    virtual void update(Vec solution, const ROMContext& context) = 0;
    virtual void reset();

protected:
    static Vec duplicate_vec(Vec source);
    void clear_vectors(std::vector<Vec>& vecs);
};

} // namespace stage2

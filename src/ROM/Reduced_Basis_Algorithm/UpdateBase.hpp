#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

class UpdateBase {
public:
    std::string name = "UpdateBase";
    int max_rank = 50;
    Eigen::MatrixXd V;
    Eigen::MatrixXd snapshots;
    Eigen::VectorXd s;
    Eigen::MatrixXd W;
    bool update_flag = false;

    explicit UpdateBase(int max_rank_in = 50) : max_rank(max_rank_in) {}
    virtual ~UpdateBase() = default;

    const Eigen::MatrixXd& getBasis(const ROMBase::ROMContext& context) const;
    void setBasis(const Eigen::MatrixXd& V_in, const Eigen::VectorXd& s_in = Eigen::VectorXd());
    void setBasis(Eigen::MatrixXd&& V_in, Eigen::VectorXd&& s_in);
    void addSnapshot(const Eigen::VectorXd& solution, const ROMBase::ROMContext& context);
    void clearSnapshots();
    virtual void update(const Eigen::VectorXd& solution, const ROMBase::ROMContext& context) = 0;
    virtual bool update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                  const Eigen::MatrixXd& V,
                                                  Eigen::MatrixXd& A_r_cache,
                                                  Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                                  Eigen::MatrixXd& AV_cache,
                                                  int& A_r_rank_cache,
                                                  ROMInfo& info) const;
    virtual void reset();
};

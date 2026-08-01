#pragma once

#include "ROM/Common/ROMTypes.hpp"

#include <Eigen/Dense>

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

    explicit UpdateBase(int max_rank_in = 50);
    virtual ~UpdateBase() = default;

    const Eigen::MatrixXd& getBasis(const ROMContext& context) const;
    void setBasis(const Eigen::MatrixXd& V_in, const Eigen::VectorXd& s_in = Eigen::VectorXd());
    void setBasis(Eigen::MatrixXd&& V_in, Eigen::VectorXd&& s_in);
    void addSnapshot(const Eigen::VectorXd& solution, const ROMContext& context);
    void clearSnapshots();
    virtual void update(const Eigen::VectorXd& solution, const ROMContext& context) = 0;
    virtual void reset();
};

#pragma once

#include <Eigen/Dense>

#include <string>

struct Analysis;
struct FEM;
struct Optimization;
struct OptimizationBase;

struct OutputData {
    double saved_Iter = 500.0;
    std::string outDir_vtk;
    std::string outDir_mat;
    std::string outDir_mat_FOM;
    std::string densityDir;
    std::string figureDir;
    int save_density_each_iter = 1;
    int save_density_interval = 1;
    int save_vtk_response_interval = 1;
    int save_vtk_opt_interval = 1;
    bool enable_live_density = true;

    mutable bool vtk_cache_ready = false;
    mutable std::string vtk_points_block;
    mutable std::string vtk_cells_block;
    mutable std::string vtk_cell_types_block;

    void prepare_case_folders() const;
    void save_density_iteration(const FEM& fem, const Eigen::MatrixXd& xPhys, int Iter) const;
    void export_response_vtk(const FEM& fem,
                             const Analysis& analysis,
                             const Eigen::MatrixXd& xPhys,
                             int Iter) const;
    void export_opt_vtk(const FEM& fem,
                        const Eigen::MatrixXd& xPhys,
                        const Eigen::MatrixXd& df_dx_mat,
                        int Iter) const;
    void save_history_csv(const Optimization& Opt) const;
    void save_rom_histories(const OptimizationBase& Opt) const;
    void initialize_vtk_cache(const FEM& fem) const;
};

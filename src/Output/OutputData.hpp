#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <string>

struct Analysis;
struct FEM;
struct Optimization;

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
    void save_rom_histories(const Optimization& Opt) const;
    void initialize_vtk_cache(const FEM& fem) const;
};

inline OutputData InitializeOutput(const myjson::json& output_json,
                                   const myjson::json& run_json,
                                   const std::string& input_json_path) {
    OutputData output;
    std::filesystem::path run_dir;
    if (output_json.contains("run_dir")) {
        run_dir = output_json.at("run_dir").get<std::string>();
    } else {
        const std::filesystem::path cfg_path = std::filesystem::path(input_json_path);
        const std::filesystem::path case_dir = cfg_path.has_parent_path() ? cfg_path.parent_path() : std::filesystem::path(".");
        run_dir = case_dir / myjson::string_value(run_json, "run_name", "FOM");
    }

    output.outDir_mat = output_json.value("outDir_mat", (run_dir / "result").string());
    output.outDir_mat_FOM = output.outDir_mat;
    const std::filesystem::path case_dir = std::filesystem::path(output.outDir_mat).parent_path();
    output.outDir_vtk = output_json.value("outDir_vtk", (case_dir / "vtk").string());
    output.densityDir = output_json.value("densityDir", (case_dir / "density").string());
    output.figureDir = output_json.value("figureDir", (case_dir / "figure").string());
    output.saved_Iter = myjson::double_value(output_json, "saved_Iter", output.saved_Iter);
    output.save_density_each_iter = myjson::bool_value(output_json, "save_density_each_iter", true) ? 1 : 0;
    output.save_density_interval = std::max(1, myjson::int_value(output_json, "save_density_interval", 1));
    output.save_vtk_response_interval = std::max(1, myjson::int_value(output_json, "save_vtk_response_interval", 1));
    output.save_vtk_opt_interval = std::max(1, myjson::int_value(output_json, "save_vtk_opt_interval", 1));
    output.enable_live_density = myjson::bool_value(output_json, "enable_live_density", true);
    return output;
}

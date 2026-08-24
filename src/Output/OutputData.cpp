#include "Output/OutputData.hpp"

#include "Optimization/Optimization.hpp"

#include "ROM/ROM.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<double> matlab_like_density_display(const FEM& fem,
                                                const Eigen::MatrixXd& xPhys) {
    if (fem.dim != 2 || xPhys.size() == 0 || fem.nelx <= 0 || fem.nely <= 0 ||
        fem.nelx * fem.nely != fem.NElem) return {};

    const int mTopo = static_cast<int>(xPhys.cols());
    std::vector<double> x_display(static_cast<std::size_t>(fem.nelx * fem.nely), 0.0);
    for (int i = 0; i < fem.nelx; ++i) {
        for (int j = 0; j < fem.nely; ++j) {
            const int e = i * fem.nely + j;
            std::vector<double> Qs(static_cast<std::size_t>(mTopo), 0.0);
            for (int v = 0; v < mTopo; ++v) Qs[static_cast<std::size_t>(v)] = std::max(1e-14, 1.0 - xPhys(e, v));
            double prod_all = 1.0;
            for (double q : Qs) prod_all *= q;
            double multi_topo = 0.0;
            for (int v = 0; v < mTopo; ++v) {
                const double prod_except = prod_all / Qs[static_cast<std::size_t>(v)];
                multi_topo += xPhys(e, v) * prod_except * static_cast<double>(v + 1);
            }
            x_display[static_cast<std::size_t>(j * fem.nelx + i)] = multi_topo;
        }
    }
    return x_display;
}

void write_density_pgm(const std::filesystem::path& path,
                       const FEM& fem,
                       const std::vector<double>& x_display) {
    std::ofstream pgm(path, std::ios::binary);
    if (!pgm) throw std::runtime_error("cannot write density pgm");
    pgm << "P5\n" << fem.nelx << " " << fem.nely << "\n255\n";
    const double max_val = std::max(1e-14, *std::max_element(x_display.begin(), x_display.end()));
    for (int j = fem.nely - 1; j >= 0; --j) {
        for (int i = 0; i < fem.nelx; ++i) {
            const double val = x_display[static_cast<std::size_t>(j * fem.nelx + i)] / max_val;
            const unsigned char pixel = static_cast<unsigned char>(std::lround(255.0 * std::clamp(val, 0.0, 1.0)));
            pgm.write(reinterpret_cast<const char*>(&pixel), 1);
        }
    }
}

void write_rom_matrix_csv(const std::string& path,
                          const std::vector<std::vector<double>>& values) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.precision(17);
    for (const auto& row : values) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            if (j) out << ",";
            out << row[j];
        }
        out << "\n";
    }
}

std::vector<std::vector<double>> rom_info_matrix(const std::vector<std::vector<ROMInfo>>& info,
                                                 const std::string& field) {
    std::vector<std::vector<double>> values(info.size());
    for (std::size_t i = 0; i < info.size(); ++i) {
        values[i].resize(info[i].size(), std::numeric_limits<double>::quiet_NaN());
        for (std::size_t j = 0; j < info[i].size(); ++j) {
            const ROMInfo& r = info[i][j];
            if (field == "rom_used") values[i][j] = r.used_rom ? 1.0 : 0.0;
            else if (field == "fom_used") values[i][j] = r.used_rom ? 0.0 : 1.0;
            else if (field == "accepted") values[i][j] = r.accepted ? 1.0 : 0.0;
            else if (field == "residual") values[i][j] = r.residual_norm;
            else if (field == "ksp_iterations") values[i][j] = static_cast<double>(r.ksp_iterations);
            else if (field == "rank") values[i][j] = static_cast<double>(r.rank);
            else if (field == "threshold") values[i][j] = r.threshold;
            else if (field == "error") values[i][j] = r.error;
            else if (field == "basis_error") values[i][j] = r.basis_error;
            else if (field == "timing_total") values[i][j] = r.timing_total;
            else if (field == "timing_petsc_to_eigen_vec") values[i][j] = r.timing_petsc_to_eigen_vec;
            else if (field == "timing_petsc_to_eigen_mat") values[i][j] = r.timing_petsc_to_eigen_mat;
            else if (field == "timing_projection") values[i][j] = r.timing_projection;
            else if (field == "timing_reduced_rhs") values[i][j] = r.timing_reduced_rhs;
            else if (field == "timing_reduced_solve") values[i][j] = r.timing_reduced_solve;
            else if (field == "timing_reconstruct") values[i][j] = r.timing_reconstruct;
            else if (field == "timing_residual") values[i][j] = r.timing_residual;
            else if (field == "timing_fom_fallback") values[i][j] = r.timing_fom_fallback;
            else if (field == "timing_fom_factorization") values[i][j] = r.timing_fom_factorization;
            else if (field == "timing_fom_solve") values[i][j] = r.timing_fom_solve;
            else if (field == "timing_basis_update") values[i][j] = r.timing_basis_update;
            else if (field == "timing_basis1_update") values[i][j] = r.timing_basis1_update;
            else if (field == "timing_online_basis_update") values[i][j] = r.timing_online_basis_update;
            else if (field == "timing_end_basis_update") values[i][j] = r.timing_end_basis_update;
            else if (field == "timing_basis_copy_reset") values[i][j] = r.timing_basis_copy_reset;
            else if (field == "timing_snapshot") values[i][j] = r.timing_snapshot;
        }
    }
    return values;
}

void save_rom_info_group(const std::string& out_dir,
                         const std::string& prefix,
                         const std::shared_ptr<ROMBase>& rom) {
    if (!rom) return;
    std::vector<std::string> fields = {
        "fom_used", "accepted", "residual", "ksp_iterations",
        "timing_total", "timing_fom_factorization", "timing_fom_solve"};
    if (rom->use_mor) {
        fields = {
            "rom_used", "fom_used", "accepted", "residual",
            "ksp_iterations", "rank", "threshold", "error", "basis_error",
            "timing_total", "timing_petsc_to_eigen_vec", "timing_petsc_to_eigen_mat",
            "timing_projection", "timing_reduced_rhs", "timing_reduced_solve",
            "timing_reconstruct", "timing_residual", "timing_fom_fallback",
            "timing_fom_factorization", "timing_fom_solve",
            "timing_basis_update", "timing_basis1_update", "timing_online_basis_update",
            "timing_end_basis_update", "timing_basis_copy_reset",
            "timing_snapshot"};
    }
    for (const std::string& field : fields) {
        write_rom_matrix_csv(out_dir + "/" + prefix + "_" + field + ".csv",
                             rom_info_matrix(rom->ROM_info, field));
    }
}

int vtk_cell_type(ElementType type) {
    if (type == ElementType::TRI3) return 5;
    if (type == ElementType::Q4) return 9;
    if (type == ElementType::T4) return 10;
    if (type == ElementType::H8) return 12;
    return 0;
}

ElementType element_type_of(const FEM& fem, int elem_id) {
    if (elem_id >= 0 && elem_id < static_cast<int>(fem.ElementTypes.size())) {
        return fem.ElementTypes[static_cast<std::size_t>(elem_id)];
    }
    if (fem.mesh_type == "2D_Q4") return ElementType::Q4;
    if (fem.mesh_type == "3D_H8") return ElementType::H8;
    return ElementType::Unknown;
}

std::string vtk_header(const std::string& title) {
    return "# vtk DataFile Version 3.0\n" + title + "\nASCII\nDATASET UNSTRUCTURED_GRID\n";
}

} // namespace

void OutputData::prepare_case_folders() const {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::filesystem::create_directories(outDir_mat);
    if (!densityDir.empty()) std::filesystem::create_directories(densityDir);
    if (!outDir_vtk.empty()) std::filesystem::create_directories(outDir_vtk);
    if (!figureDir.empty()) std::filesystem::create_directories(figureDir);
}

void OutputData::initialize_vtk_cache(const FEM& fem) const {
    if (vtk_cache_ready) return;

    std::ostringstream points;
    points.precision(17);
    points << "POINTS " << fem.NNode << " double\n";
    for (const auto& p : fem.Node) {
        points << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    std::size_t cell_int_count = 0;
    for (const auto& elem : fem.Element) cell_int_count += 1 + elem.size();

    std::ostringstream cells;
    cells << "CELLS " << fem.NElem << " " << cell_int_count << "\n";
    for (const auto& elem : fem.Element) {
        cells << elem.size();
        for (int node_id : elem) cells << " " << node_id;
        cells << "\n";
    }

    std::ostringstream cell_types;
    cell_types << "CELL_TYPES " << fem.NElem << "\n";
    for (int e = 0; e < fem.NElem; ++e) {
        cell_types << vtk_cell_type(element_type_of(fem, e)) << "\n";
    }

    vtk_points_block = points.str();
    vtk_cells_block = cells.str();
    vtk_cell_types_block = cell_types.str();
    vtk_cache_ready = true;
}

void OutputData::save_density_iteration(const FEM& fem,
                                        const Eigen::MatrixXd& xPhys,
                                        int Iter) const {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0 || xPhys.size() == 0 || !enable_live_density || fem.dim != 2) return;

    const std::filesystem::path dir = densityDir.empty()
        ? std::filesystem::path(outDir_mat) / "density"
        : std::filesystem::path(densityDir);
    std::filesystem::create_directories(dir);

    const std::vector<double> x_display = matlab_like_density_display(fem, xPhys);
    if (x_display.empty()) return;

    char name[64] = {0};
    std::snprintf(name, sizeof(name), "density_iter_%04d.pgm", Iter);
    write_density_pgm(dir / name, fem, x_display);
    write_density_pgm(std::filesystem::path(outDir_mat).parent_path() / "latest_density.pgm", fem, x_display);
}

void OutputData::export_response_vtk(const FEM& fem,
                                     const Analysis& analysis,
                                     const Eigen::MatrixXd& xPhys,
                                     int Iter) const {
    (void)Iter;
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0 || outDir_vtk.empty() || analysis.results.U.size() == 0) return;
    initialize_vtk_cache(fem);
    std::filesystem::create_directories(outDir_vtk);

    for (int step = 0; step <= analysis.Nstep; ++step) {
        char name[96] = {0};
        std::snprintf(name,
                      sizeof(name),
                      "Analysis_%d_%s_step_%d.vtk",
                      analysis.id,
                      analysis.type.c_str(),
                      step);
        std::ofstream out(std::filesystem::path(outDir_vtk) / name);
        if (!out) throw std::runtime_error("cannot write response vtk");
        out.precision(17);
        out << vtk_header("Large_scale_TO response") << vtk_points_block << vtk_cells_block << vtk_cell_types_block;
        out << "POINT_DATA " << fem.NNode << "\n";

        auto write_vector_field = [&](const std::string& field, const Eigen::MatrixXd& values) {
            if (values.size() == 0 || step >= values.cols()) return;
            out << "VECTORS " << field << " double\n";
            for (int n = 0; n < fem.NNode; ++n) {
                double v[3] = {0.0, 0.0, 0.0};
                for (int d = 0; d < fem.dim; ++d) {
                    const int dof = n * fem.dim + d;
                    if (dof < values.rows()) v[d] = values(dof, step);
                }
                out << v[0] << " " << v[1] << " " << v[2] << "\n";
            }
        };

        write_vector_field("displacement", analysis.results.U);
        write_vector_field("velocity", analysis.results.V);
        write_vector_field("acceleration", analysis.results.A);
        write_vector_field("Reaction_force", analysis.results.F_ext);

        if (xPhys.rows() == fem.NElem && xPhys.cols() > 0) {
            out << "CELL_DATA " << fem.NElem << "\n";
            if (xPhys.cols() == 1) {
                out << "SCALARS density double 1\nLOOKUP_TABLE default\n";
                for (int e = 0; e < fem.NElem; ++e) out << xPhys(e, 0) << "\n";
                out << "SCALARS Multi_Topo double 1\nLOOKUP_TABLE default\n";
                for (int e = 0; e < fem.NElem; ++e) out << xPhys(e, 0) << "\n";
            } else {
                for (int v = 0; v < xPhys.cols(); ++v) {
                    out << "SCALARS Topo_ID_" << (v + 1) << " double 1\nLOOKUP_TABLE default\n";
                    for (int e = 0; e < fem.NElem; ++e) out << xPhys(e, v) << "\n";
                }

                Eigen::VectorXd multi_topo = Eigen::VectorXd::Zero(xPhys.rows());
                for (int e = 0; e < xPhys.rows(); ++e) {
                    double prod_all = 1.0;
                    std::vector<double> q(static_cast<std::size_t>(xPhys.cols()));
                    for (int v = 0; v < xPhys.cols(); ++v) {
                        q[static_cast<std::size_t>(v)] = std::max(1e-14, 1.0 - xPhys(e, v));
                        prod_all *= q[static_cast<std::size_t>(v)];
                    }
                    for (int v = 0; v < xPhys.cols(); ++v) {
                        multi_topo(e) += xPhys(e, v) *
                            (prod_all / q[static_cast<std::size_t>(v)]) *
                            static_cast<double>(v + 1);
                    }
                }
                out << "SCALARS Multi_Topo double 1\nLOOKUP_TABLE default\n";
                for (int e = 0; e < fem.NElem; ++e) out << multi_topo(e) << "\n";
            }
        }
    }
}

void OutputData::export_opt_vtk(const FEM& fem,
                                const Eigen::MatrixXd& xPhys,
                                const Eigen::MatrixXd& df_dx_mat,
                                int Iter) const {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0 || outDir_vtk.empty() || xPhys.size() == 0) return;
    initialize_vtk_cache(fem);
    std::filesystem::create_directories(outDir_vtk);

    char name[96] = {0};
    std::snprintf(name, sizeof(name), "opt_step_%d.vtk", Iter);
    std::ofstream out(std::filesystem::path(outDir_vtk) / name);
    if (!out) throw std::runtime_error("cannot write optimization vtk");
    out.precision(17);
    out << vtk_header("Large_scale_TO optimization") << vtk_points_block << vtk_cells_block << vtk_cell_types_block;
    out << "CELL_DATA " << fem.NElem << "\n";
    const bool has_sens = df_dx_mat.rows() == xPhys.rows() && df_dx_mat.cols() == xPhys.cols();
    for (int v = 0; v < xPhys.cols(); ++v) {
        out << "SCALARS Topo_ID_" << (v + 1) << " double 1\nLOOKUP_TABLE default\n";
        for (int e = 0; e < fem.NElem; ++e) out << xPhys(e, v) << "\n";
    }
    if (has_sens) {
        for (int v = 0; v < df_dx_mat.cols(); ++v) {
            out << "SCALARS df_dx_ID_" << (v + 1) << " double 1\nLOOKUP_TABLE default\n";
            for (int e = 0; e < fem.NElem; ++e) out << df_dx_mat(e, v) << "\n";
        }
    }
    Eigen::VectorXd multi_topo = Eigen::VectorXd::Zero(xPhys.rows());
    for (int e = 0; e < xPhys.rows(); ++e) {
        double prod_all = 1.0;
        std::vector<double> q(static_cast<std::size_t>(xPhys.cols()));
        for (int v = 0; v < xPhys.cols(); ++v) {
            q[static_cast<std::size_t>(v)] = std::max(1e-14, 1.0 - xPhys(e, v));
            prod_all *= q[static_cast<std::size_t>(v)];
        }
        for (int v = 0; v < xPhys.cols(); ++v) {
            multi_topo(e) += xPhys(e, v) * (prod_all / q[static_cast<std::size_t>(v)]) * static_cast<double>(v + 1);
        }
    }
    out << "SCALARS Multi_Topo double 1\nLOOKUP_TABLE default\n";
    for (int e = 0; e < fem.NElem; ++e) out << multi_topo(e) << "\n";
}

void OutputData::save_history_csv(const Optimization& Opt) const {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::filesystem::create_directories(outDir_mat);
    std::ofstream out(outDir_mat + "/optimization_history.csv");
    out << "Iter,objective,constraint,change,rms_change,forward_time,objective_time,constraint_time,objective_sensitivity_time,constraint_sensitivity_time,mma_time,iteration_time\n";
    for (std::size_t i = 0; i < Opt.history.Iter.size(); ++i) {
        out << Opt.history.Iter[i] << ","
            << Opt.history.objective[i] << ","
            << Opt.history.constraint[i] << ","
            << Opt.history.change[i] << ","
            << Opt.history.rms_change[i] << ","
            << Opt.history.forward_time[i] << ","
            << Opt.history.objective_time[i] << ","
            << Opt.history.constraint_time[i] << ","
            << Opt.history.objective_sensitivity_time[i] << ","
            << Opt.history.constraint_sensitivity_time[i] << ","
            << Opt.history.mma_time[i] << ","
            << Opt.history.iteration_time[i] << "\n";
    }
}

void OutputData::save_rom_histories(const OptimizationBase& Opt) const {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::filesystem::create_directories(outDir_mat);
    if (!Opt.analysis_all.empty()) save_rom_info_group(outDir_mat, "forward", Opt.analysis_all.front()->ROM);
    if (!Opt.objective.groups.empty()) save_rom_info_group(outDir_mat, "obj_sens", Opt.objective.groups.front()->ROM);
}

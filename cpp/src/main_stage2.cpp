#include "Eigen_backend/DynamicElasticLinear.hpp"
#include "Eigen_backend/Optimization.hpp"
#include "MatlabLikeData.hpp"
#include "Petsc_backend/DynamicElasticLinear.hpp"
#include "Petsc_backend/Optimization.hpp"
#include "ROM.hpp"

#include <petscksp.h>

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

template <typename T>
void get_int_option(const char* name, T& value) {
    PetscInt v = static_cast<PetscInt>(value); // hãy chuyển v vềEđúng kiểu PetscInt
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetInt(nullptr, nullptr, name, &v, &set);
    if (set) value = static_cast<T>(v); // chuyển vê kiểu T
}

void get_real_option(const char* name, double& value) {
    PetscReal v = value;
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetReal(nullptr, nullptr, name, &v, &set);
    if (set) value = static_cast<double>(v);
}

void get_string_option(const char* name, std::string& value) {
    char buf[PETSC_MAX_PATH_LEN] = {0};
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, name, buf, sizeof(buf), &set);
    if (set) value = buf;
}

bool option_is_set(const char* name) {
    PetscBool set = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, name, &set);
    return set == PETSC_TRUE;
}

void set_default_option_if_missing(const char* name, const char* value) {
    if (!option_is_set(name)) PetscOptionsSetValue(nullptr, name, value);
}

void write_matrix_columns(const std::string& path, const Eigen::MatrixXd& cols) {
    if (cols.size() == 0) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.precision(17);
    for (Eigen::Index i = 0; i < cols.rows(); ++i) {
        for (Eigen::Index j = 0; j < cols.cols(); ++j) {
            if (j) out << ",";
            out << cols(i, j);
        }
        out << "\n";
    }
}

void write_info(const std::string& path,
                const FEM& fem,
                const Analysis& A,
                int mpi_size) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.precision(17);
    out << "mode=stage2_matlab_like_forward\n";
    out << "mesh_type=" << fem.mesh_type << "\n";
    out << "dim=" << fem.dim << "\n";
    out << "nnodes=" << fem.NNode << "\n";
    out << "nelem=" << fem.NElem << "\n";
    out << "ndof=" << fem.ndof << "\n";
    out << "nstep=" << A.Nstep << "\n";
    out << "dt=" << A.dt << "\n";
    out << "tmax=" << A.Tmax << "\n";
    out << "rom_name=" << (A.ROM ? A.ROM->name : "none") << "\n";
    out << "mpi_size=" << mpi_size << "\n";
    out << "assembly_time=" << A.results.assembly_time << "\n";
    out << "solve_time=" << A.results.solve_time << "\n";
    out << "ksp_iterations=";
    for (int v : A.results.ksp_iterations) out << "," << v;
    out << "\n";
    out << "ksp_residuals=";
    for (double v : A.results.ksp_residuals) out << "," << v;
    out << "\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int exit_code = 0;
    try {
        Geometry geom;
        geom.Lx = 1200.0;
        geom.Ly = 400.0;
        geom.Lz = 400.0;
        geom.nelx = 600;
        geom.nely = 200;
        geom.nelz = 1;

        std::string fem_type = "plane_stress";
        int nstep = 100;
        double tmax = 0.008;
        double x_initial = 0.5;
        double force_amp = 3333.0;
        std::string solve_mode = "ROM"; // ROM or FOM
        std::string linear_backend = "petsc"; // petsc or eigen. eigen is NP=1 forward-only for now.
        std::string method_name = "MyStrategy"; // MyStrategy or MyStrategy2
        std::string fom_solver_type = "direct"; // direct or iterative/pcg
        std::string pod_backend = "snapshot_mkl"; // snapshot_mkl, snapshot_eigen, or bdcsvd
        std::string online_backend = "incremental_mkl"; // incremental_mkl or incremental_eigen
        // Default PETSc direct-solver options if a task/launch forgets to pass them:
        //   -ksp_type: preonly(direct), cg(iterative SPD)
        //   -pc_type: cholesky(SPD direct), lu(general direct), gamg(iterative)
        //   -pc_factor_mat_solver_type: mkl_pardiso(single-node), mumps(MPI/multinode)
        set_default_option_if_missing("-ksp_type", "preonly");
        set_default_option_if_missing("-pc_type", "cholesky");
        set_default_option_if_missing("-pc_factor_mat_solver_type", "mkl_pardiso");
        std::string out_dir = "case_study/Debug";
        std::string density_dir;
        int run_optimization = 0;
        int max_iter = 5;
        double volume_limit = 0.4;
        double move = 0.1;
        double debug_fd_epsilon = 1e-6;
        int debug_fd_max_checks = 20;
        int save_density_each_iter = 1;
        int save_density_interval = 1;
        int fem_uniform = 1;
        int factor_num_threads = 1;
        int sens_element_threads = 1;
        std::string sens_type = "AVM";
        double rom_tol = 0.02;
        int rom_rank1 = 10;
        int rom_max_rank = 20;
        int rom_online = 1;
        double p_rho = 0.0;

        get_int_option("-nelx", geom.nelx);
        get_int_option("-nely", geom.nely);
        get_int_option("-nelz", geom.nelz);
        get_real_option("-Lx", geom.Lx);
        get_real_option("-Ly", geom.Ly);
        get_real_option("-Lz", geom.Lz);
        get_int_option("-nstep", nstep);
        get_real_option("-tmax", tmax);
        get_real_option("-x0", x_initial);
        get_real_option("-force_amp", force_amp);
        get_int_option("-run_optimization", run_optimization);
        get_int_option("-max_iter", max_iter);
        get_real_option("-volume_limit", volume_limit);
        get_real_option("-move", move);
        get_real_option("-debug_fd_epsilon", debug_fd_epsilon);
        get_int_option("-debug_fd_max_checks", debug_fd_max_checks);
        get_int_option("-save_density_each_iter", save_density_each_iter);
        get_int_option("-save_density_interval", save_density_interval);
        get_int_option("-density_plot_interval", save_density_interval);
        get_int_option("-fem_uniform", fem_uniform);
        get_int_option("-factor_num_threads", factor_num_threads);
        get_int_option("-sens_element_threads", sens_element_threads);
        get_real_option("-rom_tol", rom_tol);
        get_real_option("-p_rho", p_rho);
        get_int_option("-rank1", rom_rank1);
        get_int_option("-max_rank", rom_max_rank);
        get_int_option("-online", rom_online);
        get_string_option("-fem_type", fem_type);
        get_string_option("-linear_backend", linear_backend);
        get_string_option("-solve_mode", solve_mode);
        get_string_option("-method_name", method_name);
        get_string_option("-fom_solver_type", fom_solver_type);
        get_string_option("-pod_backend", pod_backend);
        get_string_option("-online_backend", online_backend);
        get_string_option("-sens_type", sens_type);
        get_string_option("-stage2_out", out_dir);
        get_string_option("-density_out", density_dir);
        if (size != 1)
        {
            if (linear_backend == "eigen" || linear_backend == "Eigen" || linear_backend == "EIGEN") {
                if (rank == 0) {
                    std::cerr << "[warning] LINEAR_BACKEND=eigen requires mpi_size=1; "
                              << "falling back to LINEAR_BACKEND=petsc for this run.\n";
                }
                linear_backend = "petsc";
            }
        }
        setenv("FACTOR_NUM_THREADS", std::to_string(std::max(1, factor_num_threads)).c_str(), 1);
        setenv("SENS_ELEMENT_THREADS", std::to_string(std::max(1, sens_element_threads)).c_str(), 1);
        if (!pod_backend.empty()) {
            setenv("STAGE2_POD_BACKEND", pod_backend.c_str(), 1);
        }
        if (!online_backend.empty()) {
            setenv("STAGE2_ONLINE_BACKEND", online_backend.c_str(), 1);
        }

        FEM fem;
        fem.type = fem_type;
        fem.nu = 0.3;
        fem.Thickness = 1.0;
        fem.uniform_element_matrices = fem_uniform != 0;
        fem = Mesh(geom, fem);
        PrecomputeFEM(fem);

        OutputData Output;
        Output.outDir_mat = out_dir;
        Output.outDir_mat_FOM = out_dir;
        Output.outDir_vtk = out_dir + "/vtk_file";
        Output.densityDir = density_dir.empty() ? out_dir + "/density" : density_dir;

        std::vector<Material> Mat;
        Mat.push_back(Material{1, "elastic", 1e-3, 0.0, 0.3, 1e-12});
        Mat.push_back(Material{2, "elastic", 210000.0, 0.0, 0.3, 6e-9});

        std::vector<Region> regions;
        Region R1;
        R1.id = 1;
        R1.mat_ids = {1, 2};
        R1.status = "design_val";
        R1.range_type = "all";
        regions.push_back(R1);
        assign_elements_to_regions(regions, fem);

        std::vector<BoundaryCondition> Dirich;
        BoundaryCondition D1;
        D1.id = 1;
        D1.kind = "fixed_disp";
        D1.flag = {1, 1, 1};
        D1.x_range = {{{0.0, 2.0}}};
        D1.y_range = {{{0.0, geom.Ly}}};
        D1.z_range = {{{-1e3, 1e3}}};
        D1.amp = 0.0;
        D1.time_type = "constant";
        Dirich.push_back(D1);

        std::vector<BoundaryCondition> Neum;
        BoundaryCondition N1;
        N1.id = 1;
        N1.kind = "nodal_force";
        N1.flag = {0, 1, 0};
        N1.x_range = {{{geom.Lx - 5.0, geom.Lx}}};
        N1.y_range = {{{190.0, 210.0}}};
        N1.z_range = {{{-1e3, 1e3}}};
        N1.amp = force_amp;
        N1.time_type = "sin";
        N1.nCycle = 1.0;
        Neum.push_back(N1);

        assign_nodes_and_dofs_to_bcs(Dirich, fem);
        assign_nodes_and_dofs_to_bcs(Neum, fem);

        Analysis A1;
        A1.id = 1;
        A1.mat_type = "elastic";
        A1.Dirich_id = {1};
        A1.Neum_id = {1};
        A1.type = "dynamic";
        A1.method = "linear";
        A1.Tmax = tmax;
        A1.Nstep = nstep;
        A1.dt = A1.Tmax / static_cast<double>(A1.Nstep);

        Interp interp;
        interp.p_E_elastic = 8.0;
        interp.p_rho = p_rho;
        interp.elastic_interp = [p = interp.p_E_elastic](const std::vector<double>& para, double x) {
            return RAMP(para, x, p);
        };
        interp.mass_density_interp = [p = interp.p_rho](const std::vector<double>& para, double x) {
            return RAMP(para, x, p);
        };

        if (solve_mode == "ROM" || solve_mode == "rom") {
            A1.ROM = petsc_backend::make_dynamic_ROM(method_name, fom_solver_type, rom_tol, rom_rank1, rom_max_rank, rom_online != 0);
        } else {
            A1.ROM = petsc_backend::make_fom_solver(fom_solver_type);
        }
        A1.rom_use_mor = solve_mode == "ROM" || solve_mode == "rom";
        A1.rom_method_name = method_name;
        A1.fom_solver_type = fom_solver_type;
        A1.rom_tol = rom_tol;
        A1.rom_rank1 = rom_rank1;
        A1.rom_max_rank = rom_max_rank;
        A1.rom_online = rom_online != 0;
        A1.solver_name = "dynamic_elastic_linear";
        A1.regions = regions;
        init_mat(Mat, A1);
        init_interp(interp, A1);
        attach_bcs_to_analysis_by_id(A1, Dirich, Neum);

        Optimization Opt;
        Opt.x = init_design_variables(fem, A1.regions, Mat, x_initial, Opt.x_label);
        Opt.MaxIter = max_iter;
        Opt.Tol = 0.0;
        Opt.filtering.rmin = 3.0;
        Opt.filtering.beta = 1.0;
        Opt.filtering.beta_loop = 50;
        Opt.filtering.beta_max = 8.0;
        Opt.filtering.T = 0.5;
        Opt.algorithm.type = "MMA";
        Opt.algorithm.move = move;
        Opt.debug_fd_epsilon = debug_fd_epsilon;
        Opt.debug_fd_max_checks = debug_fd_max_checks;
        Opt.save_density_each_iter = save_density_each_iter != 0;
        Opt.save_density_interval = std::max(1, save_density_interval);

        ObjectiveGroup ObjGroup;
        ObjGroup.analysis_id = 1;
        ObjGroup.sens_type = sens_type;
        ObjGroup.sens_function = "dynamic_elastic_base";
        ObjGroup.type = solve_mode == "ROM" || solve_mode == "rom" ? "separate_BC" : "full";
        ObjGroup.ROM = solve_mode == "ROM" || solve_mode == "rom"
            ? petsc_backend::make_dynamic_ROM(method_name, fom_solver_type, rom_tol, rom_rank1, rom_max_rank, rom_online != 0)
            : petsc_backend::make_fom_solver(fom_solver_type);
        ObjGroup.rom_use_mor = solve_mode == "ROM" || solve_mode == "rom";
        ObjGroup.rom_method_name = method_name;
        ObjGroup.fom_solver_type = fom_solver_type;
        ObjGroup.rom_tol = rom_tol;
        ObjGroup.rom_rank1 = rom_rank1;
        ObjGroup.rom_max_rank = rom_max_rank;
        ObjGroup.rom_online = rom_online != 0;
        ObjectiveTerm Term;
        Term.function_name = "L2_compliance_elastic";
        Term.weight = 1.0;
        ObjGroup.terms.push_back(Term);
        Opt.objective_groups.push_back(ObjGroup);

        Constraint Constr;
        Constr.analysis_id = 1;
        Constr.function_name = "volume_fraction";
        Constr.limit = volume_limit;
        Constr.sens_type = "AVM";
        Constr.sens_function = "volume_sens_base";
        Constr.type = "full";
        Constr.mat_id = 2;
        Constr.value_initial = 1.0;
        Opt.constraints.push_back(Constr);

        if (rank == 0) {
            std::cout << "stage2_dynamic_forward: mesh=" << fem.mesh_type
                      << " ndof=" << fem.ndof
                      << " nelem=" << fem.NElem
                      << " nstep=" << A1.Nstep
                      << " fem_uniform=" << (fem.uniform_element_matrices ? 1 : 0)
                      << " factor_threads=" << factor_num_threads
                      << " sens_element_threads=" << sens_element_threads
                      << " linear_backend=" << linear_backend
                      << " mpi_size=" << size << "\n";
        }

        std::vector<Analysis> Analysis;
        Analysis.push_back(A1);
        if (run_optimization != 0) {
            if (linear_backend == "eigen" || linear_backend == "Eigen" || linear_backend == "EIGEN") {
                eigen_backend::optimization11(fem, Analysis, Opt, Output);
            } else {
                petsc_backend::optimization11(fem, Analysis, Opt, Output);
            }
            A1 = Analysis.front();
        } else {
            if (linear_backend == "eigen" || linear_backend == "Eigen" || linear_backend == "EIGEN") {
                eigen_backend::dynamic_elastic_linear(A1, fem, Opt.x, 0);
            } else {
                petsc_backend::dynamic_elastic_linear(A1, fem, Opt.x, 0);
            }
        }

        if (rank == 0) {
            std::filesystem::create_directories(out_dir);
            write_matrix_columns(out_dir + "/U_history.csv", A1.results.U);
            write_info(out_dir + "/solver_info.txt", fem, A1, size);
            std::cout << "stage2 output: " << out_dir << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] stage2_dynamic_forward error: " << ex.what() << "\n";
        exit_code = 1;
    }

    PetscFinalize();
    return exit_code;
}

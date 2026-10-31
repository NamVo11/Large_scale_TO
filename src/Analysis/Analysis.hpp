#pragma once

#include "BounadryCondition/BoundaryCondition.hpp"
#include "FEM/FEM.hpp"
#include "Material/MaterialParas.hpp"
#include "Optimization/Interpolation/Interpolation.hpp"
#include "Region/Region.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ROMBase;
class FactorizedSolver;
class StructuralElementTotalLagrange;
struct Optimization;
struct OutputData;

struct Analysis {
    int id;
    std::string type;
    std::string solver_name;
    int Nstep;
    bool save_factorized_K_initial;

    std::shared_ptr<ROMBase> ROM;
    std::shared_ptr<Interpolation> interp;
    std::vector<std::shared_ptr<const BoundaryCondition>> Dirich;
    std::vector<std::shared_ptr<const BoundaryCondition>> Neum;
    Eigen::MatrixXd U;
    Eigen::MatrixXd F_ext;
    std::vector<int> SPCdofs;
    std::vector<int> free;
    std::vector<int> ksp_iterations;
    std::vector<double> ksp_residuals;
    double assembly_time = 0.0;
    double solve_time = 0.0;

    Analysis() = delete;

    Analysis(const myjson::json& j,
             const myjson::json& run_json,
             const std::vector<std::shared_ptr<Interpolation>>& interps,
             const std::vector<std::shared_ptr<BoundaryCondition>>& dirich_all,
             const std::vector<std::shared_ptr<BoundaryCondition>>& neum_all) {
        id = j.at("id").get<int>();
        save_factorized_K_initial = myjson::bool_value(j, "save_factorized_K_initial", false);

        const std::string requested_solver = myjson::lower_ascii(j.at("solver_name").get<std::string>());
        if (j.contains("ROM") && requested_solver != "static_geomerical_nonlinear" &&
            requested_solver != "static_geometrical_nonlinear") {
            ROM = InitializeROM(j.at("ROM"), run_json);
        }

        const int interp_id = j.at("interp_id").get<int>();
        init_interp(interps, interp_id);

        attach_bcs_by_id(j.at("Dirich_id"), j.at("Neum_id"), dirich_all, neum_all);
    }

    virtual ~Analysis() = default;
    virtual void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0);
    void init_interp(const std::vector<std::shared_ptr<Interpolation>>& interps, int interp_id);
    void attach_bcs_by_id(const myjson::json& dirich_ids,
                          const myjson::json& neum_ids,
                          const std::vector<std::shared_ptr<BoundaryCondition>>& dirich_all,
                          const std::vector<std::shared_ptr<BoundaryCondition>>& neum_all);
    std::vector<int> SPC() const {
        std::vector<int> dofs;
        for (const auto& bc : Dirich) dofs.insert(dofs.end(), bc->dof_ids.begin(), bc->dof_ids.end());
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        return dofs;
    }

    std::vector<int> free_dofs(int ndof) const {
        const std::vector<int> fixed = SPC();
        std::vector<char> is_fixed(static_cast<std::size_t>(ndof), 0);
        for (int d : fixed) {
            if (d >= 0 && d < ndof) is_fixed[static_cast<std::size_t>(d)] = 1;
        }
        std::vector<int> free;
        free.reserve(static_cast<std::size_t>(std::max(0, ndof - static_cast<int>(fixed.size()))));
        for (int d = 0; d < ndof; ++d) {
            if (!is_fixed[static_cast<std::size_t>(d)]) free.push_back(d);
        }
        return free;
    }
};

struct DynamicLinearElastic : Analysis {
    double Tmax;
    double dt;
    double alpha_M;
    double alpha_K;
    Eigen::MatrixXd V;
    Eigen::MatrixXd A;

    DynamicLinearElastic() = delete;

    DynamicLinearElastic(const myjson::json& j,
                         const myjson::json& run_json,
                         const std::vector<std::shared_ptr<Interpolation>>& interps,
                         const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                         const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : Analysis(j, run_json, interps, dirich, neum) {
        type = "dynamic";
        solver_name = "dynamic_elastic_linear";
        Tmax = j.at("Tmax").get<double>();
        Nstep = j.at("Nstep").get<int>();
        alpha_M = j.at("alpha_M").get<double>();
        alpha_K = j.at("alpha_K").get<double>();

        if (!(Tmax > 0.0)) throw std::runtime_error("dynamic_elastic_linear requires Tmax > 0");
        if (Nstep <= 0) throw std::runtime_error("dynamic_elastic_linear requires Nstep > 0");
        if (alpha_M < 0.0 || alpha_K < 0.0) {
            throw std::runtime_error("dynamic_elastic_linear requires non-negative Rayleigh damping coefficients");
        }
        dt = Tmax / static_cast<double>(Nstep);
    }
};

struct PetscDynamicLinearElastic : DynamicLinearElastic {
    Mat K_inf = nullptr;
    Mat M = nullptr;
    Mat C = nullptr;
    Mat K_eff = nullptr;
    KSP factorized_K_initial = nullptr;
    KSP factorized_K_eff = nullptr;

    PetscDynamicLinearElastic() = delete;
    PetscDynamicLinearElastic(const myjson::json& j,
                              const myjson::json& run_json,
                              const std::vector<std::shared_ptr<Interpolation>>& interps,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : DynamicLinearElastic(j, run_json, interps, dirich, neum) {}
    void assemble_K_M(const FEM& fem, const Eigen::MatrixXd& y_elem, Mat K, Mat M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct EigenDynamicLinearElastic : DynamicLinearElastic {
    Eigen::SparseMatrix<double> K_inf;
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> C;
    Eigen::SparseMatrix<double> K_eff;
    std::shared_ptr<FactorizedSolver> factorized_K_initial;
    std::shared_ptr<FactorizedSolver> factorized_K_eff;

    EigenDynamicLinearElastic() = delete;
    EigenDynamicLinearElastic(const myjson::json& j,
                              const myjson::json& run_json,
                              const std::vector<std::shared_ptr<Interpolation>>& interps,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : DynamicLinearElastic(j, run_json, interps, dirich, neum) {}
    void assemble_K_M(const FEM& fem,
                      const Eigen::MatrixXd& y_elem,
                      Eigen::SparseMatrix<double>& K,
                      Eigen::SparseMatrix<double>& M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct StaticLinearElastic : Analysis {
    StaticLinearElastic() = delete;

    StaticLinearElastic(const myjson::json& j,
                        const myjson::json& run_json,
                        const std::vector<std::shared_ptr<Interpolation>>& interps,
                        const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                        const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : Analysis(j, run_json, interps, dirich, neum) {
        type = "static";
        solver_name = "static_elastic_linear";
        Nstep = 1;
    }
};

struct PetscStaticLinearElastic : StaticLinearElastic {
    Mat K_inf = nullptr;
    KSP factorized_K_initial = nullptr;

    PetscStaticLinearElastic() = delete;
    PetscStaticLinearElastic(const myjson::json& j,
                             const myjson::json& run_json,
                             const std::vector<std::shared_ptr<Interpolation>>& interps,
                             const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                             const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : StaticLinearElastic(j, run_json, interps, dirich, neum) {}
    void assemble_K(const FEM& fem, const Eigen::MatrixXd& y_elem, Mat K) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct EigenStaticLinearElastic : StaticLinearElastic {
    Eigen::SparseMatrix<double> K_inf;
    std::shared_ptr<FactorizedSolver> factorized_K_initial;

    EigenStaticLinearElastic() = delete;
    EigenStaticLinearElastic(const myjson::json& j,
                             const myjson::json& run_json,
                             const std::vector<std::shared_ptr<Interpolation>>& interps,
                             const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                             const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : StaticLinearElastic(j, run_json, interps, dirich, neum) {}
    void assemble_K(const FEM& fem,
                    const Eigen::MatrixXd& y_elem,
                    Eigen::SparseMatrix<double>& K) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

// Full-order total-Lagrange Newton analysis. No reduced basis is used.
struct StaticGeomericalNonlinear : Analysis {
    int max_newton_iterations = 30;
    int max_line_search_iterations = 12;
    double newton_tolerance = 1e-8;
    std::vector<int> newton_iterations;
    std::vector<double> equilibrium_residuals;

    StaticGeomericalNonlinear(const myjson::json& j,
                              const myjson::json& run_json,
                              const std::vector<std::shared_ptr<Interpolation>>& interps,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                              const std::vector<std::shared_ptr<BoundaryCondition>>& neum)
        : Analysis(j, run_json, interps, dirich, neum) {
        type = "static";
        solver_name = "static_geomerical_nonlinear";
        Nstep = myjson::int_value(j, "Nstep", 10);
        max_newton_iterations = myjson::int_value(j, "max_newton_iterations", 30);
        max_line_search_iterations = myjson::int_value(j, "max_line_search_iterations", 12);
        newton_tolerance = myjson::double_value(j, "newton_tolerance", 1e-8);
        if (Nstep < 1 || max_newton_iterations < 1 ||
            max_line_search_iterations < 1 || !(newton_tolerance > 0.0))
            throw std::invalid_argument("Invalid nonlinear static solver settings");
    }
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;

private:
    struct NewtonResult {
        int iterations = 0;
        double residual_norm = 0.0;
    };

    void validate_input(const FEM& fem, const Eigen::MatrixXd& y_elem) const;
    std::vector<StructuralElementTotalLagrange*> prepare_elements(
        const FEM& fem, const Eigen::MatrixXd& y_elem) const;
    static std::vector<int> make_free_index(int ndof, const std::vector<int>& free);
    static int omp_threads_from_env();
    void set_neum(Eigen::VectorXd& f_ext, int step) const;
    void set_dirichlet_u(Eigen::VectorXd& u, int step) const;
    void assemble_system(const FEM& fem,
                         const std::vector<StructuralElementTotalLagrange*>& elements,
                         const std::vector<int>& free_index,
                         const Eigen::VectorXd& u,
                         Eigen::VectorXd& f_int,
                         Eigen::SparseMatrix<double>* K_ff);
    NewtonResult solve_newton_step(const FEM& fem,
                                   const std::vector<StructuralElementTotalLagrange*>& elements,
                                   const std::vector<int>& free_index,
                                   const Eigen::VectorXd& f_ext,
                                   Eigen::VectorXd& u,
                                   int step);
};

inline std::shared_ptr<Analysis> InitializeAnalysis(const myjson::json& analysis_json,
                                                    const myjson::json& run_json,
                                                    const std::vector<std::shared_ptr<Interpolation>>& interps,
                                                    const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                                                    const std::vector<std::shared_ptr<BoundaryCondition>>& neum) {

    const std::string linear_backend = myjson::lower_ascii(run_json.at("linear_backend").get<std::string>());
    const std::string solver_name = myjson::lower_ascii(analysis_json.at("solver_name").get<std::string>());

    std::shared_ptr<Analysis> analysis;
    if (solver_name == "dynamic_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_shared<EigenDynamicLinearElastic>(analysis_json, run_json, interps, dirich, neum);
        else analysis = std::make_shared<PetscDynamicLinearElastic>(analysis_json, run_json, interps, dirich, neum);

    } else if (solver_name == "static_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_shared<EigenStaticLinearElastic>(analysis_json, run_json, interps, dirich, neum);
        else analysis = std::make_shared<PetscStaticLinearElastic>(analysis_json, run_json, interps, dirich, neum);

    } else if (solver_name == "static_geomerical_nonlinear" ||
               solver_name == "static_geometrical_nonlinear") {
        analysis = std::make_shared<StaticGeomericalNonlinear>(analysis_json, run_json, interps, dirich, neum);

    } else if (solver_name == "dynamic_viscoelastic_linear") {
        throw std::runtime_error("dynamic_viscoelastic_linear is recognized but the Analysis class is not implemented yet");
    } else {
        throw std::runtime_error("Unsupported Analysis.solver_name: " + solver_name);
    }

    return analysis;
}

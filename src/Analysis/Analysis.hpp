#pragma once

#include "BounadryCondition/BoundaryCondition.hpp"
#include "FEM/FEM.hpp"
#include "Material/Material.hpp"
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
struct Optimization;
struct OutputData;

struct AnalysisResult {
    Eigen::MatrixXd U;
    Eigen::MatrixXd V;
    Eigen::MatrixXd A;
    Eigen::MatrixXd F_ext;
    std::vector<int> SPCdofs;
    std::vector<int> free;
    std::vector<int> ksp_iterations;
    std::vector<double> ksp_residuals;
    Mat K_inf = nullptr;
    Mat M = nullptr;
    Mat K_eff = nullptr;
    KSP factorized_K_initial = nullptr;
    KSP factorized_K_eff = nullptr;
    Eigen::SparseMatrix<double> K_inf_eigen;
    Eigen::SparseMatrix<double> M_eigen;
    Eigen::SparseMatrix<double> K_eff_eigen;
    std::shared_ptr<FactorizedSolver> factorized_K_initial_eigen;
    std::shared_ptr<FactorizedSolver> factorized_K_eff_eigen;
    double assembly_time = 0.0;
    double solve_time = 0.0;
};

struct Analysis {
    int id = 1;
    std::string mat_type = "elastic";
    std::vector<int> Dirich_id;
    std::vector<int> Neum_id;
    std::string type = "dynamic";
    std::string method = "linear";
    int Nstep = 300;
    bool save_factorized_K_initial = true;
    int interp_id = 1;

    std::shared_ptr<ROMBase> ROM;
    std::string solver_name = "dynamic_elastic_linear";
    std::shared_ptr<Interpolation> interp;
    std::vector<std::shared_ptr<const BoundaryCondition>> Dirich;
    std::vector<std::shared_ptr<const BoundaryCondition>> Neum;
    AnalysisResult results;

    Analysis() = default;

    explicit Analysis(const myjson::json& j) {
        const std::string owner = "Analysis[" + std::to_string(myjson::int_value(j, "id", 0)) + "]";
        id = myjson::required_int(j, "id", owner);
        mat_type = myjson::required_string(j, "mat_type", owner);
        myjson::require_key(j, "Dirich_id", owner);
        myjson::require_key(j, "Neum_id", owner);
        j.at("Dirich_id").get_to(Dirich_id);
        j.at("Neum_id").get_to(Neum_id);
        type = myjson::required_string(j, "type", owner);
        method = myjson::required_string(j, "method", owner);
        solver_name = myjson::required_string(j, "solver_name", owner);
        interp_id = myjson::int_value(j, "interp_id", myjson::int_value(j, "InterpType", interp_id));
        save_factorized_K_initial = myjson::bool_value(j, "save_factorized_K_initial", save_factorized_K_initial);
    }

    virtual ~Analysis() = default;
    virtual void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0);
    void init_interp(const std::vector<std::shared_ptr<Interpolation>>& interps);
    void attach_bcs_by_id(const std::vector<std::shared_ptr<BoundaryCondition>>& Dirich,
                          const std::vector<std::shared_ptr<BoundaryCondition>>& Neum);
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
    double Tmax = 0.12;
    double dt = Tmax / static_cast<double>(Nstep);

    DynamicLinearElastic() {
        type = "dynamic";
        method = "linear";
        solver_name = "dynamic_elastic_linear";
    }

    explicit DynamicLinearElastic(const myjson::json& j)
        : Analysis(j) {
        const std::string owner = "Analysis[" + std::to_string(id) + "]";
        if (myjson::lower_ascii(type) != "dynamic") {
            throw std::runtime_error("dynamic_elastic_linear requires type=dynamic");
        }
        if (myjson::lower_ascii(method) != "linear") {
            throw std::runtime_error("dynamic_elastic_linear requires method=linear");
        }
        if (myjson::lower_ascii(solver_name) != "dynamic_elastic_linear") {
            throw std::runtime_error("DynamicLinearElastic requires solver_name=dynamic_elastic_linear");
        }
        Tmax = myjson::required_double(j, "Tmax", owner);
        Nstep = myjson::required_int(j, "Nstep", owner);
        if (!(Tmax > 0.0)) throw std::runtime_error("dynamic_elastic_linear requires Tmax > 0");
        if (Nstep <= 0) throw std::runtime_error("dynamic_elastic_linear requires Nstep > 0");
        dt = Tmax / static_cast<double>(Nstep);
    }
};

struct PetscDynamicLinearElastic : DynamicLinearElastic {
    PetscDynamicLinearElastic() = default;
    explicit PetscDynamicLinearElastic(const myjson::json& j)
        : DynamicLinearElastic(j) {}
    void assemble_K_M(const FEM& fem, const Eigen::MatrixXd& y_elem, Mat K, Mat M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct EigenDynamicLinearElastic : DynamicLinearElastic {
    EigenDynamicLinearElastic() = default;
    explicit EigenDynamicLinearElastic(const myjson::json& j)
        : DynamicLinearElastic(j) {}
    void assemble_K_M(const FEM& fem,
                      const Eigen::MatrixXd& y_elem,
                      Eigen::SparseMatrix<double>& K,
                      Eigen::SparseMatrix<double>& M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct StaticLinearElastic : Analysis {
    StaticLinearElastic() {
        type = "static";
        method = "linear";
        solver_name = "static_elastic_linear";
        Nstep = 1;
    }

    explicit StaticLinearElastic(const myjson::json& j)
        : Analysis(j) {
        if (myjson::lower_ascii(type) != "static") {
            throw std::runtime_error("static_elastic_linear requires type=static");
        }
        if (myjson::lower_ascii(method) != "linear") {
            throw std::runtime_error("static_elastic_linear requires method=linear");
        }
        if (myjson::lower_ascii(solver_name) != "static_elastic_linear") {
            throw std::runtime_error("StaticLinearElastic requires solver_name=static_elastic_linear");
        }
        Nstep = 1;
    }
};

struct PetscStaticLinearElastic : StaticLinearElastic {
    PetscStaticLinearElastic() = default;
    explicit PetscStaticLinearElastic(const myjson::json& j)
        : StaticLinearElastic(j) {}
    void assemble_K(const FEM& fem, const Eigen::MatrixXd& y_elem, Mat K) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct EigenStaticLinearElastic : StaticLinearElastic {
    EigenStaticLinearElastic() = default;
    explicit EigenStaticLinearElastic(const myjson::json& j)
        : StaticLinearElastic(j) {}
    void assemble_K(const FEM& fem,
                    const Eigen::MatrixXd& y_elem,
                    Eigen::SparseMatrix<double>& K) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

inline std::unique_ptr<Analysis> InitializeAnalysis(const myjson::json& analysis_json,
                                                    const myjson::json& run_json,
                                                    const std::vector<std::shared_ptr<Interpolation>>& interps,
                                                    FEM&,
                                                    const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                                                    const std::vector<std::shared_ptr<BoundaryCondition>>& neum) {
    const std::string linear_backend = myjson::lower_ascii(myjson::string_value(run_json, "linear_backend", "petsc"));
    const std::string owner = "Analysis[" + std::to_string(myjson::int_value(analysis_json, "id", 0)) + "]";
    const std::string type = myjson::lower_ascii(myjson::required_string(analysis_json, "type", owner));
    const std::string method = myjson::lower_ascii(myjson::required_string(analysis_json, "method", owner));
    const std::string solver_name = myjson::lower_ascii(myjson::required_string(analysis_json, "solver_name", owner));

    std::unique_ptr<Analysis> analysis;
    if (type == "dynamic" && method == "linear" && solver_name == "dynamic_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_unique<EigenDynamicLinearElastic>(analysis_json);
        else analysis = std::make_unique<PetscDynamicLinearElastic>(analysis_json);
    } else if (type == "static" && method == "linear" && solver_name == "static_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_unique<EigenStaticLinearElastic>(analysis_json);
        else analysis = std::make_unique<PetscStaticLinearElastic>(analysis_json);
    } else if (type == "dynamic" && method == "linear" && solver_name == "dynamic_viscoelastic_linear") {
        throw std::runtime_error("dynamic_viscoelastic_linear is recognized but the Analysis class is not implemented yet");
    } else {
        throw std::runtime_error("Unsupported Analysis type/method/solver_name: " +
                                 type + "/" + method + "/" + solver_name);
    }

    analysis->ROM = InitializeROM(analysis_json.at("ROM"), run_json);
    analysis->init_interp(interps);
    analysis->attach_bcs_by_id(dirich, neum);
    return analysis;
}

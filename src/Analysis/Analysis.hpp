#pragma once

#include "BounadryCondition/BoundaryCondition.hpp"
#include "FEM/FEM.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class ROMBase;
class FactorizedSolver;
struct Optimization;
struct OutputData;

struct Material {
    int id = 0;
    std::string type = "elastic";
    double E_inf = 0.0;
    double E = 0.0;
    double nu = 0.3;
    double rho = 0.0;

    Material() = default;

    explicit Material(const myjson::json& j) {
        id = myjson::int_value(j, "id", id);
        type = myjson::string_value(j, "type", type);
        E_inf = myjson::double_value(j, "E_inf", E_inf);
        E = myjson::double_value(j, "E", E);
        nu = myjson::double_value(j, "nu", nu);
        rho = myjson::double_value(j, "rho", rho);
    }
};

struct MatParas {
    int nMat = 0;
    std::vector<int> mat_ids;
    std::vector<int> val_ids;
    std::vector<double> E_inf;
    std::vector<double> nu;
    std::vector<double> rho;
};

using InterpFunction = std::function<double(const std::vector<double>& para, double x)>;

struct Interp {
    double p_E_elastic = 8.0;
    double p_rho = 0.0;
    InterpFunction elastic_interp;
    InterpFunction mass_density_interp;

    Interp() = default;

    explicit Interp(const myjson::json& j) {
        p_E_elastic = myjson::double_value(j, "p_E_elastic", p_E_elastic);
        p_rho = myjson::double_value(j, "p_rho", p_rho);
    }
};

struct Region {
    int id = 0;
    std::string name;
    std::string source = "range";
    std::vector<int> mat_ids;
    std::string status = "design_val";
    std::string range_type = "all";
    std::string hmset_name;
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    std::vector<int> elem_ids;
    int nElem = 0;
    MatParas mat_paras;
    Interp interp;

    Region() = default;

    explicit Region(const myjson::json& j) {
        id = myjson::int_value(j, "id", id);
        name = myjson::string_value(j, "name", name);
        source = myjson::string_value(j, "source", source);
        status = myjson::string_value(j, "status", status);
        range_type = myjson::string_value(j, "range_type", range_type);
        hmset_name = myjson::string_value(j, "hmset_name", hmset_name);
        if (j.contains("mat_ids")) j.at("mat_ids").get_to(mat_ids);
        if (j.contains("x_range")) j.at("x_range").get_to(x_range);
        if (j.contains("y_range")) j.at("y_range").get_to(y_range);
        if (j.contains("z_range")) j.at("z_range").get_to(z_range);
    }
};

struct RangeRegion : Region {
    RangeRegion(const FEM& fem, Region region_in = Region{})
        : Region(std::move(region_in)) {
        std::vector<Region> tmp{*this};
        assign_elements_to_regions(tmp, fem);
        static_cast<Region&>(*this) = std::move(tmp.front());
    }

    static void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
};

struct HMSetRegion : Region {
    HMSetRegion(const FEM& fem, const std::string& set_name, Region region_in = Region{})
        : Region(std::move(region_in)) {
        hmset_name = set_name;
        range_type = "hmset";
        std::vector<Region> tmp{*this};
        assign_elements_to_regions(tmp, fem);
        static_cast<Region&>(*this) = std::move(tmp.front());
    }

    static void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
};

struct AnalysisResult {
    Eigen::MatrixXd U;
    Eigen::MatrixXd V;
    Eigen::MatrixXd A;
    Eigen::MatrixXd F_ext;
    Eigen::MatrixXd Lambda;
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

    std::shared_ptr<ROMBase> ROM;
    std::string solver_name = "dynamic_elastic_linear";
    std::vector<Region> regions;
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
        save_factorized_K_initial = myjson::bool_value(j, "save_factorized_K_initial", save_factorized_K_initial);
    }

    virtual ~Analysis() = default;
    virtual void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0);
    void init_mat(const std::vector<Material>& Mat);
    void init_interp(const Interp& interp);
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

double RAMP(const std::vector<double>& para, double x, double p);

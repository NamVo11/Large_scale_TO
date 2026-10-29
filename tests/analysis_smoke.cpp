#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "BounadryCondition/BoundaryCondition.hpp"
#include "Optimization/Interpolation/Interpolation.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

void check_core(ElementType type, const std::vector<std::array<double,3>>& nodes,
                const std::string& fem_type, int dim) {
    FEM fem;
    fem.dim=dim; fem.fem_type=fem_type;
    fem.Node=nodes;
    fem.Element.resize(1);
    for (std::size_t i=0;i<nodes.size();++i) fem.Element[0].push_back(static_cast<int>(i));
    fem.ElementTypes={type};
    fem.NNode=static_cast<int>(nodes.size()); fem.NElem=1; fem.ndof=dim*fem.NNode;
    fem.precompute_elements("small_strain");
    const auto regions = myjson::json::parse(
        R"([{"id":1,"source":"range","range_type":"all","status":"solid","mat_ids":[1]}])");
    const auto materials = myjson::json::parse(
        R"([{"id":1,"type":"linear_elastic","E":1.0,"nu":0.3,"rho":1.0}])");
    fem.InitializeRegions(regions);
    fem.InitializeMaterials(materials);
    const auto& element=*fem.ElementObjects.front();
    const auto small=element.compute_element_matrices(fem,0.3,1.0);
    if (small.ndof!=fem.ndof ||
        (small.ke-small.ke.transpose()).norm()>1e-10 ||
        (small.me-small.me.transpose()).norm()>1e-10 ||
        !(small.me.trace()>0.0))
        throw std::runtime_error("Small-strain element matrices invalid: "+element.name());
    for (int j=0;j<dim;++j) {
        Eigen::VectorXd rigid=Eigen::VectorXd::Zero(fem.ndof);
        for (int a=0;a<fem.NNode;++a) rigid[dim*a+j]=1;
        if ((small.ke*rigid).norm()>1e-9)
            throw std::runtime_error("Rigid translation has stiffness: "+element.name());
    }
    FEM tl_fem = fem;
    tl_fem.precompute_elements("finite_strain");
    tl_fem.InitializeMaterials(materials);
    auto& tl = dynamic_cast<StructuralElementTotalLagrange&>(*tl_fem.ElementObjects.front());
    tl.E_interp = 1.0;
    tl.nu_interp = 0.3;
    tl.rho_interp = 1.0;
    const auto X=tl.reference_coordinates(fem);
    const Eigen::VectorXd zero=Eigen::VectorXd::Zero(fem.ndof);
    const auto initial=tl.compute_response(zero,X,1.0);
    if ((initial.K_tangent-small.ke).norm()>1e-9 || initial.f_int.norm()>1e-12)
        throw std::runtime_error("TL tangent does not match small-strain limit: "+element.name());
    Eigen::VectorXd u(fem.ndof),direction(fem.ndof);
    for (int i=0;i<fem.ndof;++i) {
        u[i]=0.02*std::sin(0.7*(i+1));
        direction[i]=std::cos(0.4*(i+1));
    }
    direction.normalize();
    const auto response=tl.compute_response(u,X,1.0);
    const double h=1e-6;
    const Eigen::VectorXd fplus=tl.compute_response(u+h*direction,X,1.0,false).f_int;
    const Eigen::VectorXd fminus=tl.compute_response(u-h*direction,X,1.0,false).f_int;
    const Eigen::VectorXd finite_difference=(fplus-fminus)/(2*h);
    const double relative_error=(finite_difference-response.K_tangent*direction).norm()/
                                std::max(1.0,finite_difference.norm());
    if (relative_error>1e-7)
        throw std::runtime_error("TL tangent finite-difference check failed: "+element.name());

    const double angle=0.2;
    Eigen::VectorXd rotation=Eigen::VectorXd::Zero(fem.ndof);
    for (int a=0;a<fem.NNode;++a) {
        const double x=nodes[a][0],y=nodes[a][1];
        rotation[dim*a]=std::cos(angle)*x-std::sin(angle)*y-x;
        rotation[dim*a+1]=std::sin(angle)*x+std::cos(angle)*y-y;
    }
    if (tl.compute_response(rotation,X,1.0,false).f_int.norm()>1e-10)
        throw std::runtime_error("Rigid rotation generated internal force: "+element.name());
    std::cout << "core " << element.name() << " " << fem_type
              << " tangent_relative_error=" << relative_error << "\n";
}

void check_all_core_elements() {
    check_core(ElementType::Q4,{{0,0,0},{1,0,0},{1,1,0},{0,1,0}},"plane_stress",2);
    check_core(ElementType::Q4,{{0,0,0},{1,0,0},{1,1,0},{0,1,0}},"plane_strain",2);
    check_core(ElementType::TRI3,{{0,0,0},{1,0,0},{0,1,0}},"plane_stress",2);
    check_core(ElementType::H8,{{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                {0,0,1},{1,0,1},{1,1,1},{0,1,1}},"3d_brick",3);
    check_core(ElementType::T4,{{0,0,0},{1,0,0},{0,1,0},{0,0,1}},"3d_brick",3);
    check_core(ElementType::P6,{{0,0,0},{1,0,0},{0,1,0},
                                {0,0,1},{1,0,1},{0,1,1}},"3d_brick",3);
}

double run_case(const char* path,bool nonlinear) {
    const auto cfg=myjson::read_file(path);
    auto fem=InitializeFEM(cfg.at("FEM"));
    fem->InitializeRegions(cfg.at("Regions"));
    fem->InitializeMaterials(cfg.at("Materials"));
    const auto interps=InitializeInterpolations(cfg.at("Interp"));
    const auto dirich=InitializeBoundaryConditions(cfg.at("BoundaryConditions").at("Dirich"),*fem);
    const auto neum=InitializeBoundaryConditions(cfg.at("BoundaryConditions").at("Neum"),*fem);
    auto analysis=InitializeAnalysis(cfg.at("Analysis").at(0),cfg.at("Run"),interps,dirich,neum);
    const Eigen::MatrixXd density=Eigen::MatrixXd::Ones(fem->NElem,1);
    analysis->solver(*fem,density);
    const double max_u=analysis->U.cwiseAbs().maxCoeff();
    if (!std::isfinite(max_u) || !(max_u>0.0))
        throw std::runtime_error("Analysis produced an invalid zero/nonfinite displacement");
    if (nonlinear) {
        const auto* nl=dynamic_cast<const StaticGeomericalNonlinear*>(analysis.get());
        if (!nl || nl->equilibrium_residuals.size()!=static_cast<std::size_t>(nl->Nstep) ||
            nl->equilibrium_residuals.back()>1e-8)
            throw std::runtime_error("Nonlinear equilibrium check failed");
    } else {
        if (analysis->ksp_residuals.empty() || analysis->ksp_residuals.back()>1e-8)
            throw std::runtime_error("Linear FOM residual check failed");
    }
    std::cout << (nonlinear ? "StVenant TL" : "Elastic small strain")
              << " max_abs_u=" << max_u << "\n";
    return max_u;
}

int main(int argc,char** argv) {
    if (argc!=4) return 2;
    try {
        check_all_core_elements();
        const double linear=run_case(argv[1],false);
        const double nonlinear=run_case(argv[2],true);
        run_case(argv[3],true);
        if (std::abs(nonlinear-linear)>0.25*linear)
            throw std::runtime_error("Nonlinear and linear displacements diverge at small load");
        std::cout << "analysis smoke PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "analysis smoke FAIL: " << e.what() << "\n";
        return 1;
    }
}

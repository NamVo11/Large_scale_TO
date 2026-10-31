#include "Analysis/Analysis.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

#include <Eigen/SparseLU>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

void StaticGeomericalNonlinear::solver(const FEM& fem,const Eigen::MatrixXd& y_elem,int) {
    int mpi_initialized=0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        int ranks=1;
        MPI_Comm_size(PETSC_COMM_WORLD,&ranks);
        if (ranks!=1) throw std::runtime_error("Nonlinear Eigen FOM currently requires one MPI rank");
    }
    if (!interp) throw std::runtime_error("Nonlinear analysis requires interpolation");
    if (y_elem.rows()!=fem.NElem || y_elem.cols()<1)
        throw std::invalid_argument("Nonlinear analysis: y_elem dimensions mismatch");
    using Clock=std::chrono::steady_clock;
    assembly_time=0.0;
    solve_time=0.0;
    const int ndof=fem.ndof;
    SPCdofs=SPC();
    free=free_dofs(ndof);
    std::vector<int> free_index(ndof,-1);
    for (std::size_t i=0;i<free.size();++i) free_index[free[i]]=static_cast<int>(i);

    // Non-owning views of the elements already initialized by FEM.
    std::vector<StructuralElementTotalLagrange*> elements;

    elements.reserve(fem.NElem);

    for (int e=0;e<fem.NElem;++e) {
        auto* element=dynamic_cast<StructuralElementTotalLagrange*>(fem.ElementObjects.at(e).get());

        if (!element)
            throw std::invalid_argument("StaticGeomericalNonlinear requires FEM.element_type=finite_strain");
        if (!element->material_paras) throw std::runtime_error("Element material parameters are missing");

        const auto* paras = dynamic_cast<const ElasticParas*>(element->material_paras.get());
        if (!paras) throw std::invalid_argument("StaticGeomericalNonlinear requires elastic parameters");
        const bool is_design = myjson::lower_ascii(element->status) == "design_val";
        const int val_id = paras->val_ids.empty() ? 0 : paras->val_ids.front();
        element->E_interp = interp->calculate(paras->E_values, y_elem(e,0), is_design,
                                             val_id, interp->modulus_penalty());
        element->nu_interp = paras->nu_values.front();
        element->rho_interp = interp->calculate(paras->rho_values, y_elem(e,0), is_design,
                                               val_id, interp->mass_density_penalty());
        elements.push_back(element);
    }

    auto assemble=[&](const Eigen::VectorXd& u, Eigen::VectorXd& fint,
                      Eigen::SparseMatrix<double>* Kff) {
        const auto assembly_start=Clock::now();
        // Every trial starts from the last committed material state.
        for (auto& element:elements) element->rollback_materials();
        fint.setZero(ndof);
        std::vector<Eigen::Triplet<double>> triplets;
        if (Kff) triplets.reserve(static_cast<std::size_t>(fem.NElem)*
                                  static_cast<std::size_t>(fem.dim==2 ? 64:576));
        for (const auto& element:elements) {

            Eigen::VectorXd ue(element->dof_count());

            for (std::size_t i=0;i<element->dof_count();++i)
                ue[i]=u[static_cast<int>(element->dof_ids[i])];

            const auto response = element->create_local_matrix(
                fem,ue,Kff!=nullptr,false);
            // Lap f_int va K cua element vao global fint va Kff.
            for (std::size_t i=0;i<element->dof_count();++i) {
                const int gi=static_cast<int>(element->dof_ids[i]);
                fint[gi]+=response.f_int[i];
                if (!Kff || free_index[gi]<0) continue;
        
                for (std::size_t j=0;j<element->dof_count();++j) {
                    const int gj=static_cast<int>(element->dof_ids[j]);
                    if (free_index[gj]>=0)
                        triplets.emplace_back(free_index[gi],free_index[gj],response.ke(i,j));
                }
            }
        }
        if (Kff) {
            Kff->resize(free.size(),free.size());
            Kff->setFromTriplets(triplets.begin(),triplets.end());
            Kff->makeCompressed();
        }
        assembly_time+=std::chrono::duration<double>(Clock::now()-assembly_start).count();
    };

    U = Eigen::MatrixXd::Zero(ndof,Nstep+1);
    F_ext = Eigen::MatrixXd::Zero(ndof,Nstep+1);
    newton_iterations.clear(); 
    equilibrium_residuals.clear();
    ksp_iterations.clear();
    ksp_residuals.clear();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);

    for (int step=1;step<=Nstep;++step) {
        Eigen::VectorXd fext=Eigen::VectorXd::Zero(ndof);
        for (const auto& bc:Neum) {
            const double value=bc->value(*this,step,static_cast<double>(bc->nDof));
            for (int d:bc->dof_ids) fext[d]+=value;
        }
        for (const auto& bc:Dirich) {
            const double value=bc->value(*this,step,1.0);
            for (int d:bc->dof_ids) u[d]=value;
        }
        F_ext.col(step) = fext;
        double initial_norm=0.0;
        bool converged=false;
        int updates=0;
        double final_norm=0.0;
        for (int iter=0;iter<=max_newton_iterations;++iter) {
            Eigen::VectorXd fint;
            Eigen::SparseMatrix<double> Kff;
            assemble(u,fint,&Kff);
            Eigen::VectorXd residual(free.size());
            for (std::size_t i=0;i<free.size();++i)
                residual[i]=fext[free[i]]-fint[free[i]];
            final_norm=residual.norm();
            if (iter==0) initial_norm=final_norm;
            const double scale=std::max({1.0,fext.norm(),initial_norm});
            if (final_norm<=newton_tolerance*scale) { converged=true; break; }
            if (iter==max_newton_iterations)
                break;
            const auto solve_start=Clock::now();
            Eigen::SparseLU<Eigen::SparseMatrix<double>> linear_solver;
            linear_solver.compute(Kff);
            if (linear_solver.info()!=Eigen::Success)
                throw std::runtime_error("Nonlinear Newton tangent factorization failed");
            const Eigen::VectorXd du=linear_solver.solve(residual);
            if (linear_solver.info()!=Eigen::Success || !du.allFinite())
                throw std::runtime_error("Nonlinear Newton tangent solve failed");
            solve_time+=std::chrono::duration<double>(Clock::now()-solve_start).count();
            bool accepted=false;
            double alpha=1.0;
            for (int ls=0;ls<max_line_search_iterations;++ls,alpha*=0.5) {
                Eigen::VectorXd trial=u;
                for (std::size_t i=0;i<free.size();++i) trial[free[i]]+=alpha*du[i];
                try {
                    Eigen::VectorXd trial_fint;
                    assemble(trial,trial_fint,nullptr);
                    double trial_norm2=0.0;
                    for (int d:free) {
                        const double r=fext[d]-trial_fint[d];
                        trial_norm2+=r*r;
                    }
                    if (std::sqrt(trial_norm2)<=(1.0-1e-4*alpha)*final_norm ||
                        std::sqrt(trial_norm2)<=newton_tolerance*scale) {
                        u.swap(trial);
                        accepted=true;
                        break;
                    }
                } catch (const std::runtime_error&) {
                    // An inverted trial deformation is rejected by the line search.
                }
            }
            if (!accepted) break;
            ++updates;
        }
        if (!converged) {
            for (auto& element:elements) element->rollback_materials();
            throw std::runtime_error("Nonlinear static Newton failed at load step "+
                                     std::to_string(step)+", residual="+std::to_string(final_norm));
        }
        for (auto& element:elements) element->commit_materials();
        U.col(step)=u;
        newton_iterations.push_back(updates);
        equilibrium_residuals.push_back(final_norm);
        ksp_iterations.push_back(updates);
        ksp_residuals.push_back(final_norm);
        std::cout << "nonlinear step=" << step << " newton=" << updates
                  << " residual=" << final_norm << "\n";
    }
}

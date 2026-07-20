
    std::vector<std::vector<double>> Debug(const FEM &fem,
                                           Analysis &analysis,
                                           const std::vector<std::vector<double>> &x,
                                           const std::vector<int> &x_label,
                                           ObjectiveGroup &object,
                                           const Filtering &filtering,
                                           const FilteringState &filtering_state,
                                           const OutputData &output,
                                           double fd_epsilon,
                                           int max_checks,
                                           int Iter)
    {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        std::vector<std::vector<double>> avm = AVM(fem, analysis, x, x_label, object, filtering, filtering_state, Iter);

        const int nelem = fem.NElem;
        const int nvar = nvar_of(x);
        std::vector<std::pair<int, int>> candidates;
        candidates.reserve(static_cast<std::size_t>(nelem * nvar));
        for (int v = 0; v < nvar; ++v)
        {
            for (int e = 0; e < nelem; ++e)
            {
                if (x_label[static_cast<std::size_t>(e)] == 0)
                    candidates.push_back({e, v});
            }
        }
        if (max_checks <= 0 || max_checks > static_cast<int>(candidates.size()))
            max_checks = static_cast<int>(candidates.size());
        const int stride = std::max(1, static_cast<int>(candidates.size()) / std::max(1, max_checks));

        std::ofstream out;
        if (rank == 0)
        {
            std::filesystem::create_directories(output.outDir_mat);
            out.open(output.outDir_mat + "/sensitivity_debug.csv");
            out.precision(17);
            out << "Iter,elem,var,x,avm,fd,abs_err,rel_err,obj_plus,obj_minus\n";
            std::cout << "Debug sensitivity: finite-difference check on " << max_checks
                      << " design variables, eps=" << fd_epsilon << "\n";
        }

        int checked = 0;
        for (int idx = 0; idx < static_cast<int>(candidates.size()) && checked < max_checks; idx += stride)
        {
            const int e = candidates[static_cast<std::size_t>(idx)].first;
            const int v = candidates[static_cast<std::size_t>(idx)].second;
            const double x0 = x[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
            const double h_plus = std::min(fd_epsilon, 1.0 - x0);
            const double h_minus = std::min(fd_epsilon, x0);
            if (h_plus <= 0.0 && h_minus <= 0.0)
                continue;

            double obj_plus = object.value;
            double obj_minus = object.value;
            if (h_plus > 0.0)
            {
                std::vector<std::vector<double>> xp = x;
                FilteringState state_plus = filtering_state;
                xp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = x0 + h_plus;
                filtering_calculation(xp, x_label, filtering, state_plus);
                dynamic_elastic_linear(analysis, fem, state_plus.xPhys, Iter);
                obj_plus = objective_group_value_for_debug(fem, analysis, state_plus.xPhys, object);
            }
            if (h_minus > 0.0)
            {
                std::vector<std::vector<double>> xm = x;
                FilteringState state_minus = filtering_state;
                xm[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = x0 - h_minus;
                filtering_calculation(xm, x_label, filtering, state_minus);
                dynamic_elastic_linear(analysis, fem, state_minus.xPhys, Iter);
                obj_minus = objective_group_value_for_debug(fem, analysis, state_minus.xPhys, object);
            }

            double fd = 0.0;
            if (h_plus > 0.0 && h_minus > 0.0)
            {
                fd = (obj_plus - obj_minus) / (h_plus + h_minus);
            }
            else if (h_plus > 0.0)
            {
                fd = (obj_plus - object.value) / h_plus;
            }
            else
            {
                fd = (object.value - obj_minus) / h_minus;
            }

            if (rank == 0)
            {
                const double g = avm[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
                const double abs_err = std::abs(g - fd);
                const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                out << Iter << "," << e << "," << v << "," << x0 << ","
                    << g << "," << fd << "," << abs_err << "," << rel_err << ","
                    << obj_plus << "," << obj_minus << "\n";
            }
            ++checked;
        }

        dynamic_elastic_linear(analysis, fem, filtering_state.xPhys, Iter);
        return avm;
    }

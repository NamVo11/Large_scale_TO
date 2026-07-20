
    std::vector<std::vector<double>> AVM(const FEM &fem,
                                         Analysis &analysis,
                                         const std::vector<std::vector<double>> &,
                                         const std::vector<int> &x_label,
                                         ObjectiveGroup &object,
                                         const Filtering &filtering,
                                         const FilteringState &filtering_state,
                                         int Iter)
    {
        std::vector<std::vector<double>> out = dynamic_elastic_base_full_impl(fem, analysis, filtering_state.xPhys, object, filtering, filtering_state, Iter);
        for (std::size_t e = 0; e < out.size(); ++e)
        {
            if (x_label[e] != 0)
                std::fill(out[e].begin(), out[e].end(), 0.0);
        }
        return out;
    }

    std::vector<std::vector<double>> AVM(const FEM &fem,
                                         Analysis &analysis,
                                         const std::vector<std::vector<double>> &,
                                         const std::vector<int> &x_label,
                                         Constraint &object,
                                         const Filtering &filtering,
                                         const FilteringState &filtering_state,
                                         int Iter)
    {
        (void)fem;
        (void)analysis;
        (void)Iter;
        std::vector<std::vector<double>> dfdx_phys = object.d_explicit_df_dx;
        std::vector<std::vector<double>> out = filtering_sensitivity(dfdx_phys, filtering, filtering_state);
        for (std::size_t e = 0; e < out.size(); ++e)
        {
            if (x_label[e] != 0)
                std::fill(out[e].begin(), out[e].end(), 0.0);
        }
        return out;
    }
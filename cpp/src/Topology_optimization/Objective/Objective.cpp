 void evaluate_objective(const FEM &fem,
                            std::vector<Analysis> &analysis_all,
                            const std::vector<std::vector<double>> &xPhys,
                            std::vector<ObjectiveGroup> &objective_groups,
                            double &obj_value)
    {
        obj_value = 0.0;
        for (auto &group : objective_groups)
        {
            Analysis &analysis = analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
            group.value = 0.0;
            group.d_explicit_dfdu_all.clear();
            group.d_explicit_dfdv_all.clear();
            group.d_explicit_dfda_all.clear();
            group.d_explicit_dfdFext_all.clear();
            group.d_explicit_df_dx.clear();
            for (auto &term : group.terms)
            {
                if (lower(term.function_name) != "l2_compliance_elastic")
                    throw std::runtime_error("Unknown objective term");
                L2_compliance_elastic(fem, analysis, xPhys, term);
                if (term.value_initial == 0.0)
                    term.value_initial = term.value;
                const double alpha = term.weight / term.value_initial;
                group.value += alpha * term.value;
                add_scaled(group.d_explicit_dfdu_all, term.dfdu_all, alpha);
                add_scaled(group.d_explicit_dfdv_all, term.dfdv_all, alpha);
                add_scaled(group.d_explicit_dfda_all, term.dfda_all, alpha);
                add_scaled(group.d_explicit_dfdFext_all, term.dfdFext_all, alpha);
                add_scaled(group.d_explicit_df_dx, term.df_dx, alpha);
            }
            obj_value += group.value;
        }
    }
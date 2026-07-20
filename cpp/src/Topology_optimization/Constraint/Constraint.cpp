 void evaluate_constraint(const FEM &fem,
                             std::vector<Analysis> &analysis_all,
                             const std::vector<std::vector<double>> &xPhys,
                             std::vector<Constraint> &constraints,
                             std::vector<double> &constr_value_vec)
    {
        constr_value_vec.assign(constraints.size(), 0.0);
        for (std::size_t i = 0; i < constraints.size(); ++i)
        {
            Constraint &c = constraints[i];
            Analysis &analysis = analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
            if (lower(c.function_name) != "volume_fraction")
                throw std::runtime_error("Unknown constraint");
            volume_fraction(fem, analysis, xPhys, c);
            if (c.value_initial == 0.0)
                c.value_initial = c.value;
            constr_value_vec[i] = c.value / c.value_initial - c.limit;
            const double alpha = 1.0 / c.value_initial;
            for (auto &row : c.d_explicit_df_dx)
            {
                for (double &v : row)
                    v *= alpha;
            }
        }
    }
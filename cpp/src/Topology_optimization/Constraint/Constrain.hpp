struct Constraint {
    int analysis_id = 1;
    std::string function_name;
    double limit = 0.0;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    int mat_id = 0;
    double value_initial = 0.0;
    double value = 0.0;
    std::vector<std::vector<double>> d_explicit_dfdu_all;
    std::vector<std::vector<double>> d_explicit_dfdv_all;
    std::vector<std::vector<double>> d_explicit_dfda_all;
    std::vector<std::vector<double>> d_explicit_dfdFext_all;
    std::vector<std::vector<double>> d_explicit_df_dx;
};
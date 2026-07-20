struct ObjectiveGroup {
    int analysis_id = 1;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    std::shared_ptr<ROMBase> ROM;
    std::vector<ObjectiveTerm> terms;
    double value = 0.0;
    std::vector<std::vector<double>> d_explicit_dfdu_all;
    std::vector<std::vector<double>> d_explicit_dfdv_all;
    std::vector<std::vector<double>> d_explicit_dfda_all;
    std::vector<std::vector<double>> d_explicit_dfdFext_all;
    std::vector<std::vector<double>> d_explicit_df_dx;
};

struct ObjectiveTerm {
    std::string function_name;
    double weight = 1.0;
    double value_initial = 0.0;
    double value = 0.0;
    std::vector<std::vector<double>> dfdu_all;
    std::vector<std::vector<double>> dfdv_all;
    std::vector<std::vector<double>> dfda_all;
    std::vector<std::vector<double>> dfdFext_all;
    std::vector<std::vector<double>> df_dx;
};
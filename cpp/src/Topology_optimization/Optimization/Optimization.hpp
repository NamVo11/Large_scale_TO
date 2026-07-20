struct Optimization {
    std::vector<std::vector<double>> x0;
    std::vector<std::vector<double>> x;
    std::vector<int> x_label;
    int MaxIter = 300;
    double Tol = 0.0;
    bool plot_during_optimization = false;
    Filtering filtering;
    Algorithm algorithm;
    double debug_fd_epsilon = 1e-6;
    int debug_fd_max_checks = 20;
    bool save_density_each_iter = true;
    int save_density_interval = 1;
    std::vector<ObjectiveGroup> objective_groups;
    std::vector<Constraint> constraints;
    OptimizationHistory history;
};

struct OptimizationHistory {
    std::vector<int> Iter;
    std::vector<double> objective;
    std::vector<double> constraint;
    std::vector<double> change;
    std::vector<double> rms_change;
    std::vector<double> forward_time;
    std::vector<double> objective_time;
    std::vector<double> constraint_time;
    std::vector<double> objective_sensitivity_time;
    std::vector<double> constraint_sensitivity_time;
    std::vector<double> mma_time;
    std::vector<double> iteration_time;
};
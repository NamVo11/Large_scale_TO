
struct BoundaryCondition {
    int id = 0;
    std::string kind;
    std::array<int, 3> flag{0, 0, 0};
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;
    double amp = 0.0;
    std::string time_type = "constant";
    double nCycle = 0.0;
    double power = 0.0;
    double phase = 0.0;

    std::vector<int> node_ids;
    std::vector<int> dof_ids;
    int nNode = 0;
    int nDof = 0;
    std::function<double(int)> fun_value;
};

void assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem);
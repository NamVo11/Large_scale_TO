struct Material {
    int id = 0;
    std::string type = "elastic";
    double E_inf = 0.0;
    double E = 0.0;
    double nu = 0.3;
    double rho = 0.0;
};


struct MatParas {
    int nMat = 0;
    std::vector<int> mat_ids;
    std::vector<int> val_ids;
    std::vector<double> E_inf;
    std::vector<double> nu;
    std::vector<double> rho;
};

struct Analysis {
    int id = 1;
    std::string mat_type = "elastic";
    std::vector<int> Dirich_id;
    std::vector<int> Neum_id;
    std::string type = "dynamic";
    std::string method = "linear";
    double Tmax = 0.12;
    int Nstep = 300;
    double dt = Tmax / static_cast<double>(Nstep);

    std::shared_ptr<ROMBase> ROM;
    std::string solver_name = "dynamic_elastic_linear";
    std::vector<Region> regions;
    std::vector<BoundaryCondition> Dirich;
    std::vector<BoundaryCondition> Neum;
    AnalysisResult results;
};

void attach_bcs_to_analysis_by_id(Analysis& A,
                                  const std::vector<BoundaryCondition>& Dirich,
                                  const std::vector<BoundaryCondition>& Neum);

void init_mat(const std::vector<Material>& Mat, Analysis& A);

void init_interp(const Interp& interp, Analysis& A);


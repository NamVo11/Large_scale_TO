void attach_bcs_to_analysis_by_id(Analysis& A,
                                  const std::vector<BoundaryCondition>& Dirich,
                                  const std::vector<BoundaryCondition>& Neum) {
    A.Dirich.clear();
    A.Neum.clear();
    for (int idx : ids_by_id(Dirich, A.Dirich_id)) {
        BoundaryCondition bc = Dirich[static_cast<std::size_t>(idx)];
        bc.fun_value = [bc, &A](int istep) { return bc_value(bc, A, istep); };
        A.Dirich.push_back(std::move(bc));
    }
    for (int idx : ids_by_id(Neum, A.Neum_id)) {
        BoundaryCondition bc = Neum[static_cast<std::size_t>(idx)];
        bc.fun_value = [bc, &A](int istep) { return bc_value(bc, A, istep); };
        A.Neum.push_back(std::move(bc));
    }
}

void init_mat(const std::vector<Material>& Mat, Analysis& A) {
    if (lower(A.mat_type) != "elastic") throw std::runtime_error("Only elastic init_mat is implemented in stage2");
    for (auto& region : A.regions) {
        std::vector<int> mat_ids = region.mat_ids;
        if (lower(region.status) == "void") mat_ids = {1};
        region.mat_paras.nMat = static_cast<int>(mat_ids.size());
        region.mat_paras.mat_ids = mat_ids;
        region.mat_paras.val_ids.clear();
        region.mat_paras.E_inf.clear();
        region.mat_paras.nu.clear();
        region.mat_paras.rho.clear();
        for (int id : mat_ids) {
            auto it = std::find_if(Mat.begin(), Mat.end(), [id](const Material& m) { return m.id == id; });
            if (it == Mat.end()) throw std::runtime_error("Material id not found");
            region.mat_paras.E_inf.push_back(it->E_inf != 0.0 ? it->E_inf : it->E);
            region.mat_paras.nu.push_back(it->nu);
            region.mat_paras.rho.push_back(it->rho);
        }
        if (lower(region.status) == "design_val") {
            for (std::size_t i = 1; i < mat_ids.size(); ++i) region.mat_paras.val_ids.push_back(mat_ids[i] - 1);
        } else {
            region.mat_paras.val_ids.push_back(std::max(0, mat_ids.front() - 1));
        }
    }
}

void init_interp(const Interp& interp, Analysis& A) {
    for (auto& region : A.regions) {
        if (lower(region.status) == "design_val") {
            region.interp = interp;
        } else {
            region.interp.elastic_interp = [](const std::vector<double>& para, double) { return para.front(); };
            region.interp.mass_density_interp = [](const std::vector<double>& para, double) { return para.front(); };
        }
    }
}
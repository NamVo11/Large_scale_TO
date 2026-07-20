std::vector<std::vector<double>> init_design_variables(const FEM& fem,
                                                       const std::vector<Region>&,
                                                       const std::vector<Material>& Mat,
                                                       double x0,
                                                       std::vector<int>& x_label) {
    const int nvar = std::max(1, static_cast<int>(Mat.size()) - 1);
    std::vector<std::vector<double>> x(static_cast<std::size_t>(fem.NElem), std::vector<double>(nvar, x0));
    x_label.assign(static_cast<std::size_t>(fem.NElem), 0);
    return x;
}
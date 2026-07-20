
struct Region {
    int id = 0;
    std::vector<int> mat_ids;
    std::string status = "design_val";
    std::string range_type = "all";
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    std::vector<int> elem_ids;
    int nElem = 0;
    MatParas mat_paras;
    Interp interp;
};

void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
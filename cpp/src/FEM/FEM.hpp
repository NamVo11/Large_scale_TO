
struct Geometry {
    double Lx = 1200.0;
    double Ly = 400.0;
    double Lz = 120.0;
    int nelx = 120;
    int nely = 40;
    int nelz = 1;
};

struct FEM {
    std::string type = "plane_stress";
    double nu = 0.3;
    double Thickness = 1.0;

    std::vector<std::array<double, 3>> Node;
    std::vector<std::vector<int>> Element;
    std::vector<std::vector<PetscInt>> ElementDofs;
    std::vector<std::array<double, 3>> elem_center;
    int dim = 2;
    std::string mesh_type = "2D_Q4";
    bool uniform_element_matrices = true;
    int NNode = 0;
    int NElem = 0;
    int ndof = 0;
    std::vector<int> AllDofs;
    int nelx = 0;
    int nely = 0;
    int nelz = 0;
};

FEM Mesh(const Geometry& geom, FEM fem);

void PrecomputeFEM(FEM& fem);
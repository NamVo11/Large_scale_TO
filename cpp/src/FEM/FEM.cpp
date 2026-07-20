
FEM Mesh(const Geometry& geom, FEM fem) {
    const std::string type = lower(fem.type);
    fem.Node.clear();
    fem.Element.clear();
    fem.ElementDofs.clear();
    fem.elem_center.clear();
    fem.nelx = geom.nelx;
    fem.nely = geom.nely;
    fem.nelz = geom.nelz;

    if (type == "plane_stress" || type == "plane_strain") {
        fem.dim = 2;
        fem.mesh_type = "2D_Q4";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        auto nid = [ny](int i, int j) { return i * ny + j; };

        fem.Node.reserve(static_cast<std::size_t>(nx * ny));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                fem.Node.push_back({i * ex, j * ey, 0.0});
            }
        }

        fem.Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        fem.elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        for (int i = 0; i < geom.nelx; ++i) {
            for (int j = 0; j < geom.nely; ++j) {
                std::vector<int> elem{nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)};
                fem.Element.push_back(elem);
                fem.elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, 0.0});
            }
        }
    } else if (type == "3d_brick") {
        fem.dim = 3;
        fem.mesh_type = "3D_H8";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const double ez = geom.Lz / static_cast<double>(geom.nelz);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        const int nz = geom.nelz + 1;
        auto nid = [nx, ny](int i, int j, int k) { return k * nx * ny + i * ny + j; };

        fem.Node.reserve(static_cast<std::size_t>(nx * ny * nz));
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    fem.Node.push_back({i * ex, j * ey, k * ez});
                }
            }
        }

        fem.Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        fem.elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        for (int k = 0; k < geom.nelz; ++k) {
            for (int i = 0; i < geom.nelx; ++i) {
                for (int j = 0; j < geom.nely; ++j) {
                    fem.Element.push_back({
                        nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                        nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)});
                    fem.elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, (k + 0.5) * ez});
                }
            }
        }
    } else {
        throw std::runtime_error("Mesh: unsupported fem.type");
    }

    fem.NNode = static_cast<int>(fem.Node.size());
    fem.NElem = static_cast<int>(fem.Element.size());
    fem.ndof = fem.NNode * fem.dim;
    fem.AllDofs.resize(static_cast<std::size_t>(fem.ndof));
    std::iota(fem.AllDofs.begin(), fem.AllDofs.end(), 0);
    return fem;
}

void PrecomputeFEM(FEM& fem) {
    if (fem.mesh_type != "2D_Q4" && fem.mesh_type != "3D_H8") {
        throw std::runtime_error("PrecomputeFEM: unsupported mesh type");
    }

    fem.ElementDofs.resize(static_cast<std::size_t>(fem.NElem));
    for (int e = 0; e < fem.NElem; ++e) {
        const auto& elem = fem.Element.at(static_cast<std::size_t>(e));
        auto& dofs = fem.ElementDofs[static_cast<std::size_t>(e)];
        dofs.resize(elem.size() * static_cast<std::size_t>(fem.dim));
        for (std::size_t a = 0; a < elem.size(); ++a) {
            for (int d = 0; d < fem.dim; ++d) {
                dofs[a * static_cast<std::size_t>(fem.dim) + static_cast<std::size_t>(d)] =
                    static_cast<PetscInt>(fem.dim * elem[a] + d);
            }
        }
    }
}

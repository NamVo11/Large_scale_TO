
    void filtering_initilization(const FEM &fem, Filtering &filtering, FilteringState &state)
    {
        const int r = std::max(0, static_cast<int>(std::ceil(filtering.rmin)) - 1);
        state.neighbors.assign(static_cast<std::size_t>(fem.NElem), {});
        state.weights.assign(static_cast<std::size_t>(fem.NElem), {});
        state.Hs.assign(static_cast<std::size_t>(fem.NElem), 0.0);
        auto eid2 = [&fem](int i, int j)
        { return i * fem.nely + j; };
        auto eid3 = [&fem](int i, int j, int k)
        { return k * fem.nelx * fem.nely + i * fem.nely + j; };
        if (fem.dim == 2)
        {
            for (int i1 = 0; i1 < fem.nelx; ++i1)
            {
                for (int j1 = 0; j1 < fem.nely; ++j1)
                {
                    const int e1 = eid2(i1, j1);
                    for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2)
                    {
                        for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2)
                        {
                            const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2));
                            const double w = filtering.rmin - dist;
                            if (w > 0.0)
                            {
                                state.neighbors[static_cast<std::size_t>(e1)].push_back(eid2(i2, j2));
                                state.weights[static_cast<std::size_t>(e1)].push_back(w);
                                state.Hs[static_cast<std::size_t>(e1)] += w;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (int k1 = 0; k1 < fem.nelz; ++k1)
            {
                for (int i1 = 0; i1 < fem.nelx; ++i1)
                {
                    for (int j1 = 0; j1 < fem.nely; ++j1)
                    {
                        const int e1 = eid3(i1, j1, k1);
                        for (int k2 = std::max(0, k1 - r); k2 <= std::min(fem.nelz - 1, k1 + r); ++k2)
                        {
                            for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2)
                            {
                                for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2)
                                {
                                    const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2));
                                    const double w = filtering.rmin - dist;
                                    if (w > 0.0)
                                    {
                                        state.neighbors[static_cast<std::size_t>(e1)].push_back(eid3(i2, j2, k2));
                                        state.weights[static_cast<std::size_t>(e1)].push_back(w);
                                        state.Hs[static_cast<std::size_t>(e1)] += w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void filtering_calculation(const std::vector<std::vector<double>> &x,
                               const std::vector<int> &x_label,
                               const Filtering &filtering,
                               FilteringState &state)
    {
        const int nelem = static_cast<int>(x.size());
        const int nvar = nvar_of(x);
        state.xTilde = zeros(nelem, nvar);
        state.xPhys = zeros(nelem, nvar);
        const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
        for (int e = 0; e < nelem; ++e)
        {
            for (std::size_t p = 0; p < state.neighbors[static_cast<std::size_t>(e)].size(); ++p)
            {
                const int j = state.neighbors[static_cast<std::size_t>(e)][p];
                const double w = state.weights[static_cast<std::size_t>(e)][p];
                for (int v = 0; v < nvar; ++v)
                    state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] += w * x[static_cast<std::size_t>(j)][static_cast<std::size_t>(v)];
            }
            for (int v = 0; v < nvar; ++v)
            {
                state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] /= state.Hs[static_cast<std::size_t>(e)];
                state.xPhys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                    (std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] - filtering.T))) / denom;
            }
            if (x_label[static_cast<std::size_t>(e)] > 0)
                std::fill(state.xPhys[static_cast<std::size_t>(e)].begin(), state.xPhys[static_cast<std::size_t>(e)].end(), 1.0);
            if (x_label[static_cast<std::size_t>(e)] < 0)
                std::fill(state.xPhys[static_cast<std::size_t>(e)].begin(), state.xPhys[static_cast<std::size_t>(e)].end(), 0.0);
        }
    }

    std::vector<std::vector<double>> filtering_sensitivity(const std::vector<std::vector<double>> &df_dx_phys,
                                                           const Filtering &filtering,
                                                           const FilteringState &state)
    {
        const int nelem = static_cast<int>(df_dx_phys.size());
        const int nvar = nvar_of(df_dx_phys);
        std::vector<std::vector<double>> tmp = zeros(nelem, nvar);
        const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
        for (int e = 0; e < nelem; ++e)
        {
            for (int v = 0; v < nvar; ++v)
            {
                const double dproj = filtering.beta * sech2(filtering.beta * (state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] - filtering.T)) / denom;
                tmp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = df_dx_phys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] * dproj / state.Hs[static_cast<std::size_t>(e)];
            }
        }
        std::vector<std::vector<double>> out = zeros(nelem, nvar);
        for (int e = 0; e < nelem; ++e)
        {
            for (std::size_t p = 0; p < state.neighbors[static_cast<std::size_t>(e)].size(); ++p)
            {
                const int j = state.neighbors[static_cast<std::size_t>(e)][p];
                const double w = state.weights[static_cast<std::size_t>(e)][p];
                for (int v = 0; v < nvar; ++v)
                    out[static_cast<std::size_t>(j)][static_cast<std::size_t>(v)] += w * tmp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
            }
        }
        return out;
    }


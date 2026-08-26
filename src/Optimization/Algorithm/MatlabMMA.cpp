#include "Optimization/Algorithm/Algorithm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
    double dot(const std::vector<double> &a, const std::vector<double> &b)
    {
        double v = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            v += a[i] * b[i];
        return v;
    }

    double l2_norm(const std::vector<double> &a)
    {
        return std::sqrt(dot(a, a));
    }

    double max_abs(const std::vector<double> &a)
    {
        double v = 0.0;
        for (double x : a)
            v = std::max(v, std::abs(x));
        return v;
    }

    double max_value(const std::vector<double> &a)
    {
        return *std::max_element(a.begin(), a.end());
    }

    std::vector<double> solve_dense(std::vector<double> A, std::vector<double> b, int n)
    {
        const double tiny = 1.0e-30;
        for (int k = 0; k < n; ++k)
        {
            int pivot = k;
            double pivot_abs = std::abs(A[static_cast<std::size_t>(k * n + k)]);
            for (int i = k + 1; i < n; ++i)
            {
                const double cand = std::abs(A[static_cast<std::size_t>(i * n + k)]);
                if (cand > pivot_abs)
                {
                    pivot_abs = cand;
                    pivot = i;
                }
            }
            if (pivot_abs < tiny)
                throw std::runtime_error("MatlabMMA dense solve failed: singular matrix");
            if (pivot != k)
            {
                for (int j = k; j < n; ++j)
                    std::swap(A[static_cast<std::size_t>(k * n + j)], A[static_cast<std::size_t>(pivot * n + j)]);
                std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
            }
            for (int i = k + 1; i < n; ++i)
            {
                const double factor = A[static_cast<std::size_t>(i * n + k)] / A[static_cast<std::size_t>(k * n + k)];
                A[static_cast<std::size_t>(i * n + k)] = 0.0;
                for (int j = k + 1; j < n; ++j)
                    A[static_cast<std::size_t>(i * n + j)] -= factor * A[static_cast<std::size_t>(k * n + j)];
                b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
            }
        }

        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        for (int i = n - 1; i >= 0; --i)
        {
            double rhs = b[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < n; ++j)
                rhs -= A[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(j)];
            x[static_cast<std::size_t>(i)] = rhs / A[static_cast<std::size_t>(i * n + i)];
        }
        return x;
    }

    struct SubsolvResult
    {
        std::vector<double> x;
        std::vector<double> y;
        double z = 0.0;
        std::vector<double> lam;
        std::vector<double> xsi;
        std::vector<double> eta;
        std::vector<double> mu;
        double zet = 0.0;
        std::vector<double> s;
    };

    SubsolvResult subsolv_matlab(int m,
                                 int n,
                                 double epsimin,
                                 const std::vector<double> &low,
                                 const std::vector<double> &upp,
                                 const std::vector<double> &alfa,
                                 const std::vector<double> &beta,
                                 const std::vector<double> &p0,
                                 const std::vector<double> &q0,
                                 const std::vector<double> &P,
                                 const std::vector<double> &Q,
                                 double a0,
                                 const std::vector<double> &a,
                                 const std::vector<double> &b,
                                 const std::vector<double> &c,
                                 const std::vector<double> &d)
    {
        const double inf_guard = std::numeric_limits<double>::infinity();
        (void)inf_guard;

        double epsi = 1.0;
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        std::vector<double> y(static_cast<std::size_t>(m), 1.0);
        double z = 1.0;
        std::vector<double> lam(static_cast<std::size_t>(m), 1.0);
        std::vector<double> xsi(static_cast<std::size_t>(n), 0.0);
        std::vector<double> eta(static_cast<std::size_t>(n), 0.0);
        std::vector<double> mu(static_cast<std::size_t>(m), 0.0);
        double zet = 1.0;
        std::vector<double> s(static_cast<std::size_t>(m), 1.0);

        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = 0.5 * (alfa[static_cast<std::size_t>(i)] + beta[static_cast<std::size_t>(i)]);
            xsi[static_cast<std::size_t>(i)] = std::max(1.0, 1.0 / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]));
            eta[static_cast<std::size_t>(i)] = std::max(1.0, 1.0 / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]));
        }
        for (int j = 0; j < m; ++j)
            mu[static_cast<std::size_t>(j)] = std::max(1.0, 0.5 * c[static_cast<std::size_t>(j)]);

        while (epsi > epsimin)
        {
            std::vector<double> ux1(static_cast<std::size_t>(n)), xl1(static_cast<std::size_t>(n));
            std::vector<double> ux2(static_cast<std::size_t>(n)), xl2(static_cast<std::size_t>(n));
            std::vector<double> uxinv1(static_cast<std::size_t>(n)), xlinv1(static_cast<std::size_t>(n));
            std::vector<double> plam(static_cast<std::size_t>(n)), qlam(static_cast<std::size_t>(n));
            std::vector<double> gvec(static_cast<std::size_t>(m), 0.0);
            std::vector<double> dpsidx(static_cast<std::size_t>(n), 0.0);

            auto compute_residual = [&]() {
                for (int i = 0; i < n; ++i)
                {
                    ux1[static_cast<std::size_t>(i)] = upp[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)];
                    xl1[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - low[static_cast<std::size_t>(i)];
                    ux2[static_cast<std::size_t>(i)] = ux1[static_cast<std::size_t>(i)] * ux1[static_cast<std::size_t>(i)];
                    xl2[static_cast<std::size_t>(i)] = xl1[static_cast<std::size_t>(i)] * xl1[static_cast<std::size_t>(i)];
                    uxinv1[static_cast<std::size_t>(i)] = 1.0 / ux1[static_cast<std::size_t>(i)];
                    xlinv1[static_cast<std::size_t>(i)] = 1.0 / xl1[static_cast<std::size_t>(i)];
                    plam[static_cast<std::size_t>(i)] = p0[static_cast<std::size_t>(i)];
                    qlam[static_cast<std::size_t>(i)] = q0[static_cast<std::size_t>(i)];
                }
                std::fill(gvec.begin(), gvec.end(), 0.0);
                for (int j = 0; j < m; ++j)
                {
                    for (int i = 0; i < n; ++i)
                    {
                        const std::size_t ji = static_cast<std::size_t>(j * n + i);
                        plam[static_cast<std::size_t>(i)] += P[ji] * lam[static_cast<std::size_t>(j)];
                        qlam[static_cast<std::size_t>(i)] += Q[ji] * lam[static_cast<std::size_t>(j)];
                        gvec[static_cast<std::size_t>(j)] += P[ji] * uxinv1[static_cast<std::size_t>(i)] + Q[ji] * xlinv1[static_cast<std::size_t>(i)];
                    }
                }
                for (int i = 0; i < n; ++i)
                    dpsidx[static_cast<std::size_t>(i)] = plam[static_cast<std::size_t>(i)] / ux2[static_cast<std::size_t>(i)] -
                                                          qlam[static_cast<std::size_t>(i)] / xl2[static_cast<std::size_t>(i)];

                std::vector<double> residu;
                residu.reserve(static_cast<std::size_t>(2 * n + 4 * m + 2));
                for (int i = 0; i < n; ++i)
                    residu.push_back(dpsidx[static_cast<std::size_t>(i)] - xsi[static_cast<std::size_t>(i)] + eta[static_cast<std::size_t>(i)]);
                for (int j = 0; j < m; ++j)
                    residu.push_back(c[static_cast<std::size_t>(j)] + d[static_cast<std::size_t>(j)] * y[static_cast<std::size_t>(j)] - mu[static_cast<std::size_t>(j)] - lam[static_cast<std::size_t>(j)]);
                residu.push_back(a0 - zet - dot(a, lam));
                for (int j = 0; j < m; ++j)
                    residu.push_back(gvec[static_cast<std::size_t>(j)] - a[static_cast<std::size_t>(j)] * z - y[static_cast<std::size_t>(j)] + s[static_cast<std::size_t>(j)] - b[static_cast<std::size_t>(j)]);
                for (int i = 0; i < n; ++i)
                    residu.push_back(xsi[static_cast<std::size_t>(i)] * (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]) - epsi);
                for (int i = 0; i < n; ++i)
                    residu.push_back(eta[static_cast<std::size_t>(i)] * (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]) - epsi);
                for (int j = 0; j < m; ++j)
                    residu.push_back(mu[static_cast<std::size_t>(j)] * y[static_cast<std::size_t>(j)] - epsi);
                residu.push_back(zet * z - epsi);
                for (int j = 0; j < m; ++j)
                    residu.push_back(lam[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)] - epsi);
                return residu;
            };

            std::vector<double> residu = compute_residual();
            double residunorm = l2_norm(residu);
            double residumax = max_abs(residu);
            int ittt = 0;

            while (residumax > 0.9 * epsi && ittt < 200)
            {
                ++ittt;

                std::vector<double> ux3(static_cast<std::size_t>(n)), xl3(static_cast<std::size_t>(n));
                std::vector<double> uxinv2(static_cast<std::size_t>(n)), xlinv2(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i)
                {
                    ux1[static_cast<std::size_t>(i)] = upp[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)];
                    xl1[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - low[static_cast<std::size_t>(i)];
                    ux2[static_cast<std::size_t>(i)] = ux1[static_cast<std::size_t>(i)] * ux1[static_cast<std::size_t>(i)];
                    xl2[static_cast<std::size_t>(i)] = xl1[static_cast<std::size_t>(i)] * xl1[static_cast<std::size_t>(i)];
                    ux3[static_cast<std::size_t>(i)] = ux1[static_cast<std::size_t>(i)] * ux2[static_cast<std::size_t>(i)];
                    xl3[static_cast<std::size_t>(i)] = xl1[static_cast<std::size_t>(i)] * xl2[static_cast<std::size_t>(i)];
                    uxinv1[static_cast<std::size_t>(i)] = 1.0 / ux1[static_cast<std::size_t>(i)];
                    xlinv1[static_cast<std::size_t>(i)] = 1.0 / xl1[static_cast<std::size_t>(i)];
                    uxinv2[static_cast<std::size_t>(i)] = 1.0 / ux2[static_cast<std::size_t>(i)];
                    xlinv2[static_cast<std::size_t>(i)] = 1.0 / xl2[static_cast<std::size_t>(i)];
                    plam[static_cast<std::size_t>(i)] = p0[static_cast<std::size_t>(i)];
                    qlam[static_cast<std::size_t>(i)] = q0[static_cast<std::size_t>(i)];
                }
                std::fill(gvec.begin(), gvec.end(), 0.0);
                std::vector<double> GG(static_cast<std::size_t>(m * n), 0.0);
                for (int j = 0; j < m; ++j)
                {
                    for (int i = 0; i < n; ++i)
                    {
                        const std::size_t ji = static_cast<std::size_t>(j * n + i);
                        plam[static_cast<std::size_t>(i)] += P[ji] * lam[static_cast<std::size_t>(j)];
                        qlam[static_cast<std::size_t>(i)] += Q[ji] * lam[static_cast<std::size_t>(j)];
                        gvec[static_cast<std::size_t>(j)] += P[ji] * uxinv1[static_cast<std::size_t>(i)] + Q[ji] * xlinv1[static_cast<std::size_t>(i)];
                        GG[ji] = P[ji] * uxinv2[static_cast<std::size_t>(i)] - Q[ji] * xlinv2[static_cast<std::size_t>(i)];
                    }
                }

                std::vector<double> delx(static_cast<std::size_t>(n)), dely(static_cast<std::size_t>(m)), dellam(static_cast<std::size_t>(m));
                std::vector<double> diagx(static_cast<std::size_t>(n)), diagxinv(static_cast<std::size_t>(n));
                std::vector<double> diagy(static_cast<std::size_t>(m)), diagyinv(static_cast<std::size_t>(m));
                std::vector<double> diaglam(static_cast<std::size_t>(m)), diaglamyi(static_cast<std::size_t>(m));
                for (int i = 0; i < n; ++i)
                {
                    dpsidx[static_cast<std::size_t>(i)] = plam[static_cast<std::size_t>(i)] / ux2[static_cast<std::size_t>(i)] -
                                                          qlam[static_cast<std::size_t>(i)] / xl2[static_cast<std::size_t>(i)];
                    delx[static_cast<std::size_t>(i)] = dpsidx[static_cast<std::size_t>(i)] -
                                                        epsi / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]) +
                                                        epsi / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]);
                    diagx[static_cast<std::size_t>(i)] = 2.0 * (plam[static_cast<std::size_t>(i)] / ux3[static_cast<std::size_t>(i)] +
                                                                qlam[static_cast<std::size_t>(i)] / xl3[static_cast<std::size_t>(i)]) +
                                                        xsi[static_cast<std::size_t>(i)] / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]) +
                                                        eta[static_cast<std::size_t>(i)] / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]);
                    diagxinv[static_cast<std::size_t>(i)] = 1.0 / diagx[static_cast<std::size_t>(i)];
                }

                const double delz = a0 - dot(a, lam) - epsi / z;
                for (int j = 0; j < m; ++j)
                {
                    dely[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j)] + d[static_cast<std::size_t>(j)] * y[static_cast<std::size_t>(j)] -
                                                        lam[static_cast<std::size_t>(j)] - epsi / y[static_cast<std::size_t>(j)];
                    dellam[static_cast<std::size_t>(j)] = gvec[static_cast<std::size_t>(j)] - a[static_cast<std::size_t>(j)] * z -
                                                          y[static_cast<std::size_t>(j)] - b[static_cast<std::size_t>(j)] +
                                                          epsi / lam[static_cast<std::size_t>(j)];
                    diagy[static_cast<std::size_t>(j)] = d[static_cast<std::size_t>(j)] + mu[static_cast<std::size_t>(j)] / y[static_cast<std::size_t>(j)];
                    diagyinv[static_cast<std::size_t>(j)] = 1.0 / diagy[static_cast<std::size_t>(j)];
                    diaglam[static_cast<std::size_t>(j)] = s[static_cast<std::size_t>(j)] / lam[static_cast<std::size_t>(j)];
                    diaglamyi[static_cast<std::size_t>(j)] = diaglam[static_cast<std::size_t>(j)] + diagyinv[static_cast<std::size_t>(j)];
                }

                std::vector<double> dx(static_cast<std::size_t>(n), 0.0);
                std::vector<double> dlam(static_cast<std::size_t>(m), 0.0);
                double dz = 0.0;

                if (m < n)
                {
                    std::vector<double> blam(static_cast<std::size_t>(m), 0.0);
                    for (int j = 0; j < m; ++j)
                    {
                        blam[static_cast<std::size_t>(j)] = dellam[static_cast<std::size_t>(j)] + dely[static_cast<std::size_t>(j)] / diagy[static_cast<std::size_t>(j)];
                        for (int i = 0; i < n; ++i)
                            blam[static_cast<std::size_t>(j)] -= GG[static_cast<std::size_t>(j * n + i)] * delx[static_cast<std::size_t>(i)] / diagx[static_cast<std::size_t>(i)];
                    }

                    const int sz = m + 1;
                    std::vector<double> AA(static_cast<std::size_t>(sz * sz), 0.0);
                    std::vector<double> bb(static_cast<std::size_t>(sz), 0.0);
                    for (int j = 0; j < m; ++j)
                    {
                        for (int k = 0; k < m; ++k)
                        {
                            double val = (j == k) ? diaglamyi[static_cast<std::size_t>(j)] : 0.0;
                            for (int i = 0; i < n; ++i)
                                val += GG[static_cast<std::size_t>(j * n + i)] * diagxinv[static_cast<std::size_t>(i)] * GG[static_cast<std::size_t>(k * n + i)];
                            AA[static_cast<std::size_t>(j * sz + k)] = val;
                        }
                        AA[static_cast<std::size_t>(j * sz + m)] = a[static_cast<std::size_t>(j)];
                        AA[static_cast<std::size_t>(m * sz + j)] = a[static_cast<std::size_t>(j)];
                        bb[static_cast<std::size_t>(j)] = blam[static_cast<std::size_t>(j)];
                    }
                    AA[static_cast<std::size_t>(m * sz + m)] = -zet / z;
                    bb[static_cast<std::size_t>(m)] = delz;

                    const std::vector<double> solut = solve_dense(AA, bb, sz);
                    for (int j = 0; j < m; ++j)
                        dlam[static_cast<std::size_t>(j)] = solut[static_cast<std::size_t>(j)];
                    dz = solut[static_cast<std::size_t>(m)];
                    for (int i = 0; i < n; ++i)
                    {
                        double gtdlam = 0.0;
                        for (int j = 0; j < m; ++j)
                            gtdlam += GG[static_cast<std::size_t>(j * n + i)] * dlam[static_cast<std::size_t>(j)];
                        dx[static_cast<std::size_t>(i)] = -delx[static_cast<std::size_t>(i)] / diagx[static_cast<std::size_t>(i)] -
                                                          gtdlam / diagx[static_cast<std::size_t>(i)];
                    }
                }
                else
                {
                    std::vector<double> diaglamyiinv(static_cast<std::size_t>(m), 0.0);
                    std::vector<double> dellamyi(static_cast<std::size_t>(m), 0.0);
                    for (int j = 0; j < m; ++j)
                    {
                        diaglamyiinv[static_cast<std::size_t>(j)] = 1.0 / diaglamyi[static_cast<std::size_t>(j)];
                        dellamyi[static_cast<std::size_t>(j)] = dellam[static_cast<std::size_t>(j)] + dely[static_cast<std::size_t>(j)] / diagy[static_cast<std::size_t>(j)];
                    }
                    const int sz = n + 1;
                    std::vector<double> AA(static_cast<std::size_t>(sz * sz), 0.0);
                    std::vector<double> bb(static_cast<std::size_t>(sz), 0.0);
                    for (int i = 0; i < n; ++i)
                    {
                        for (int k = 0; k < n; ++k)
                        {
                            double val = (i == k) ? diagx[static_cast<std::size_t>(i)] : 0.0;
                            for (int j = 0; j < m; ++j)
                                val += GG[static_cast<std::size_t>(j * n + i)] * diaglamyiinv[static_cast<std::size_t>(j)] * GG[static_cast<std::size_t>(j * n + k)];
                            AA[static_cast<std::size_t>(i * sz + k)] = val;
                        }
                    }
                    double azz = zet / z;
                    for (int j = 0; j < m; ++j)
                        azz += a[static_cast<std::size_t>(j)] * a[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)];
                    for (int i = 0; i < n; ++i)
                    {
                        double axz = 0.0;
                        double bx = delx[static_cast<std::size_t>(i)];
                        for (int j = 0; j < m; ++j)
                        {
                            axz -= GG[static_cast<std::size_t>(j * n + i)] * a[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)];
                            bx += GG[static_cast<std::size_t>(j * n + i)] * dellamyi[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)];
                        }
                        AA[static_cast<std::size_t>(i * sz + n)] = axz;
                        AA[static_cast<std::size_t>(n * sz + i)] = axz;
                        bb[static_cast<std::size_t>(i)] = -bx;
                    }
                    double bz = delz;
                    for (int j = 0; j < m; ++j)
                        bz -= a[static_cast<std::size_t>(j)] * dellamyi[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)];
                    AA[static_cast<std::size_t>(n * sz + n)] = azz;
                    bb[static_cast<std::size_t>(n)] = -bz;

                    const std::vector<double> solut = solve_dense(AA, bb, sz);
                    for (int i = 0; i < n; ++i)
                        dx[static_cast<std::size_t>(i)] = solut[static_cast<std::size_t>(i)];
                    dz = solut[static_cast<std::size_t>(n)];
                    for (int j = 0; j < m; ++j)
                    {
                        double ggdx = 0.0;
                        for (int i = 0; i < n; ++i)
                            ggdx += GG[static_cast<std::size_t>(j * n + i)] * dx[static_cast<std::size_t>(i)];
                        dlam[static_cast<std::size_t>(j)] = ggdx / diaglamyi[static_cast<std::size_t>(j)] -
                                                            dz * a[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)] +
                                                            dellamyi[static_cast<std::size_t>(j)] / diaglamyi[static_cast<std::size_t>(j)];
                    }
                }

                std::vector<double> dy(static_cast<std::size_t>(m));
                std::vector<double> dxsi(static_cast<std::size_t>(n)), deta(static_cast<std::size_t>(n));
                std::vector<double> dmu(static_cast<std::size_t>(m)), ds(static_cast<std::size_t>(m));
                for (int j = 0; j < m; ++j)
                {
                    dy[static_cast<std::size_t>(j)] = -dely[static_cast<std::size_t>(j)] / diagy[static_cast<std::size_t>(j)] +
                                                      dlam[static_cast<std::size_t>(j)] / diagy[static_cast<std::size_t>(j)];
                    dmu[static_cast<std::size_t>(j)] = -mu[static_cast<std::size_t>(j)] + epsi / y[static_cast<std::size_t>(j)] -
                                                       mu[static_cast<std::size_t>(j)] * dy[static_cast<std::size_t>(j)] / y[static_cast<std::size_t>(j)];
                    ds[static_cast<std::size_t>(j)] = -s[static_cast<std::size_t>(j)] + epsi / lam[static_cast<std::size_t>(j)] -
                                                      s[static_cast<std::size_t>(j)] * dlam[static_cast<std::size_t>(j)] / lam[static_cast<std::size_t>(j)];
                }
                for (int i = 0; i < n; ++i)
                {
                    dxsi[static_cast<std::size_t>(i)] = -xsi[static_cast<std::size_t>(i)] + epsi / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]) -
                                                        xsi[static_cast<std::size_t>(i)] * dx[static_cast<std::size_t>(i)] / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]);
                    deta[static_cast<std::size_t>(i)] = -eta[static_cast<std::size_t>(i)] + epsi / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]) +
                                                        eta[static_cast<std::size_t>(i)] * dx[static_cast<std::size_t>(i)] / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]);
                }
                const double dzet = -zet + epsi / z - zet * dz / z;

                std::vector<double> xx;
                std::vector<double> dxx;
                xx.reserve(static_cast<std::size_t>(n * 2 + m * 4 + 2));
                dxx.reserve(xx.capacity());
                for (int j = 0; j < m; ++j)
                {
                    xx.push_back(y[static_cast<std::size_t>(j)]);
                    dxx.push_back(dy[static_cast<std::size_t>(j)]);
                }
                xx.push_back(z);
                dxx.push_back(dz);
                for (int j = 0; j < m; ++j)
                {
                    xx.push_back(lam[static_cast<std::size_t>(j)]);
                    dxx.push_back(dlam[static_cast<std::size_t>(j)]);
                }
                for (int i = 0; i < n; ++i)
                {
                    xx.push_back(xsi[static_cast<std::size_t>(i)]);
                    dxx.push_back(dxsi[static_cast<std::size_t>(i)]);
                }
                for (int i = 0; i < n; ++i)
                {
                    xx.push_back(eta[static_cast<std::size_t>(i)]);
                    dxx.push_back(deta[static_cast<std::size_t>(i)]);
                }
                for (int j = 0; j < m; ++j)
                {
                    xx.push_back(mu[static_cast<std::size_t>(j)]);
                    dxx.push_back(dmu[static_cast<std::size_t>(j)]);
                }
                xx.push_back(zet);
                dxx.push_back(dzet);
                for (int j = 0; j < m; ++j)
                {
                    xx.push_back(s[static_cast<std::size_t>(j)]);
                    dxx.push_back(ds[static_cast<std::size_t>(j)]);
                }

                std::vector<double> stepxx(xx.size(), 0.0);
                for (std::size_t i = 0; i < xx.size(); ++i)
                    stepxx[i] = -1.01 * dxx[i] / xx[i];
                double stmxx = max_value(stepxx);
                double stmalfa = -std::numeric_limits<double>::infinity();
                double stmbeta = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < n; ++i)
                {
                    stmalfa = std::max(stmalfa, -1.01 * dx[static_cast<std::size_t>(i)] / (x[static_cast<std::size_t>(i)] - alfa[static_cast<std::size_t>(i)]));
                    stmbeta = std::max(stmbeta, 1.01 * dx[static_cast<std::size_t>(i)] / (beta[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]));
                }
                double steg = 1.0 / std::max({std::max(stmalfa, stmbeta), stmxx, 1.0});

                const std::vector<double> xold = x;
                const std::vector<double> yold = y;
                const double zold = z;
                const std::vector<double> lamold = lam;
                const std::vector<double> xsiold = xsi;
                const std::vector<double> etaold = eta;
                const std::vector<double> muold = mu;
                const double zetold = zet;
                const std::vector<double> sold = s;

                int itto = 0;
                double resinew = 2.0 * residunorm;
                while (resinew > residunorm && itto < 50)
                {
                    ++itto;
                    for (int i = 0; i < n; ++i)
                    {
                        x[static_cast<std::size_t>(i)] = xold[static_cast<std::size_t>(i)] + steg * dx[static_cast<std::size_t>(i)];
                        xsi[static_cast<std::size_t>(i)] = xsiold[static_cast<std::size_t>(i)] + steg * dxsi[static_cast<std::size_t>(i)];
                        eta[static_cast<std::size_t>(i)] = etaold[static_cast<std::size_t>(i)] + steg * deta[static_cast<std::size_t>(i)];
                    }
                    for (int j = 0; j < m; ++j)
                    {
                        y[static_cast<std::size_t>(j)] = yold[static_cast<std::size_t>(j)] + steg * dy[static_cast<std::size_t>(j)];
                        lam[static_cast<std::size_t>(j)] = lamold[static_cast<std::size_t>(j)] + steg * dlam[static_cast<std::size_t>(j)];
                        mu[static_cast<std::size_t>(j)] = muold[static_cast<std::size_t>(j)] + steg * dmu[static_cast<std::size_t>(j)];
                        s[static_cast<std::size_t>(j)] = sold[static_cast<std::size_t>(j)] + steg * ds[static_cast<std::size_t>(j)];
                    }
                    z = zold + steg * dz;
                    zet = zetold + steg * dzet;
                    residu = compute_residual();
                    resinew = l2_norm(residu);
                    steg *= 0.5;
                }
                residunorm = resinew;
                residumax = max_abs(residu);
                steg *= 2.0;
            }
            epsi *= 0.1;
        }

        SubsolvResult result;
        result.x = std::move(x);
        result.y = std::move(y);
        result.z = z;
        result.lam = std::move(lam);
        result.xsi = std::move(xsi);
        result.eta = std::move(eta);
        result.mu = std::move(mu);
        result.zet = zet;
        result.s = std::move(s);
        return result;
    }
}

void MatlabMMA::Reset()
{
    iter_ = 0;
    std::fill(xold1_.begin(), xold1_.end(), 0.0);
    std::fill(xold2_.begin(), xold2_.end(), 0.0);
    std::fill(low_.begin(), low_.end(), 0.0);
    std::fill(upp_.begin(), upp_.end(), 0.0);
}

void MatlabMMA::Update(double *xval,
                       const double *df0dx,
                       const double *fval,
                       const double *dfdx,
                       const double *xmin,
                       const double *xmax)
{
    ++iter_;

    constexpr double epsimin = 1.0e-7;
    constexpr double raa0 = 0.00001;
    constexpr double move = 0.5;
    constexpr double albefa = 0.1;
    constexpr double asyinit = 0.5;
    constexpr double asyincr = 1.2;
    constexpr double asydecr = 0.7;
    constexpr double xmamieps_value = 0.00001;

    std::vector<double> xold_current(static_cast<std::size_t>(n_));
    for (int i = 0; i < n_; ++i)
        xold_current[static_cast<std::size_t>(i)] = xval[i];

    if (iter_ < 3)
    {
        for (int i = 0; i < n_; ++i)
        {
            low_[static_cast<std::size_t>(i)] = xval[i] - asyinit * (xmax[i] - xmin[i]);
            upp_[static_cast<std::size_t>(i)] = xval[i] + asyinit * (xmax[i] - xmin[i]);
        }
    }
    else
    {
        for (int i = 0; i < n_; ++i)
        {
            const double zzz = (xval[i] - xold1_[static_cast<std::size_t>(i)]) *
                               (xold1_[static_cast<std::size_t>(i)] - xold2_[static_cast<std::size_t>(i)]);
            double factor = 1.0;
            if (zzz > 0.0)
                factor = asyincr;
            else if (zzz < 0.0)
                factor = asydecr;

            low_[static_cast<std::size_t>(i)] = xval[i] - factor * (xold1_[static_cast<std::size_t>(i)] - low_[static_cast<std::size_t>(i)]);
            upp_[static_cast<std::size_t>(i)] = xval[i] + factor * (upp_[static_cast<std::size_t>(i)] - xold1_[static_cast<std::size_t>(i)]);
            low_[static_cast<std::size_t>(i)] = std::max(low_[static_cast<std::size_t>(i)], xval[i] - 10.0 * (xmax[i] - xmin[i]));
            low_[static_cast<std::size_t>(i)] = std::min(low_[static_cast<std::size_t>(i)], xval[i] - 0.01 * (xmax[i] - xmin[i]));
            upp_[static_cast<std::size_t>(i)] = std::min(upp_[static_cast<std::size_t>(i)], xval[i] + 10.0 * (xmax[i] - xmin[i]));
            upp_[static_cast<std::size_t>(i)] = std::max(upp_[static_cast<std::size_t>(i)], xval[i] + 0.01 * (xmax[i] - xmin[i]));
        }
    }

    std::vector<double> alfa(static_cast<std::size_t>(n_));
    std::vector<double> beta(static_cast<std::size_t>(n_));
    std::vector<double> xmami(static_cast<std::size_t>(n_));
    std::vector<double> ux1(static_cast<std::size_t>(n_));
    std::vector<double> xl1(static_cast<std::size_t>(n_));
    std::vector<double> ux2(static_cast<std::size_t>(n_));
    std::vector<double> xl2(static_cast<std::size_t>(n_));
    std::vector<double> p0(static_cast<std::size_t>(n_));
    std::vector<double> q0(static_cast<std::size_t>(n_));
    std::vector<double> P(static_cast<std::size_t>(m_ * n_), 0.0);
    std::vector<double> Q(static_cast<std::size_t>(m_ * n_), 0.0);
    std::vector<double> b(static_cast<std::size_t>(m_), 0.0);

    for (int i = 0; i < n_; ++i)
    {
        const double zzz1a = low_[static_cast<std::size_t>(i)] + albefa * (xval[i] - low_[static_cast<std::size_t>(i)]);
        const double zzz2a = xval[i] - move * (xmax[i] - xmin[i]);
        alfa[static_cast<std::size_t>(i)] = std::max(std::max(zzz1a, zzz2a), xmin[i]);

        const double zzz1b = upp_[static_cast<std::size_t>(i)] - albefa * (upp_[static_cast<std::size_t>(i)] - xval[i]);
        const double zzz2b = xval[i] + move * (xmax[i] - xmin[i]);
        beta[static_cast<std::size_t>(i)] = std::min(std::min(zzz1b, zzz2b), xmax[i]);

        xmami[static_cast<std::size_t>(i)] = std::max(xmax[i] - xmin[i], xmamieps_value);
        ux1[static_cast<std::size_t>(i)] = upp_[static_cast<std::size_t>(i)] - xval[i];
        xl1[static_cast<std::size_t>(i)] = xval[i] - low_[static_cast<std::size_t>(i)];
        ux2[static_cast<std::size_t>(i)] = ux1[static_cast<std::size_t>(i)] * ux1[static_cast<std::size_t>(i)];
        xl2[static_cast<std::size_t>(i)] = xl1[static_cast<std::size_t>(i)] * xl1[static_cast<std::size_t>(i)];

        const double p0base = std::max(df0dx[i], 0.0);
        const double q0base = std::max(-df0dx[i], 0.0);
        const double pq0 = 0.001 * (p0base + q0base) + raa0 / xmami[static_cast<std::size_t>(i)];
        p0[static_cast<std::size_t>(i)] = (p0base + pq0) * ux2[static_cast<std::size_t>(i)];
        q0[static_cast<std::size_t>(i)] = (q0base + pq0) * xl2[static_cast<std::size_t>(i)];
    }

    for (int j = 0; j < m_; ++j)
    {
        b[static_cast<std::size_t>(j)] = -fval[j];
        for (int i = 0; i < n_; ++i)
        {
            const double grad = dfdx[static_cast<std::size_t>(j * n_ + i)];
            const double pbase = std::max(grad, 0.0);
            const double qbase = std::max(-grad, 0.0);
            const double pq = 0.001 * (pbase + qbase) + raa0 / xmami[static_cast<std::size_t>(i)];
            const std::size_t ji = static_cast<std::size_t>(j * n_ + i);
            P[ji] = (pbase + pq) * ux2[static_cast<std::size_t>(i)];
            Q[ji] = (qbase + pq) * xl2[static_cast<std::size_t>(i)];
            b[static_cast<std::size_t>(j)] += P[ji] / ux1[static_cast<std::size_t>(i)] + Q[ji] / xl1[static_cast<std::size_t>(i)];
        }
    }

    SubsolvResult result = subsolv_matlab(m_, n_, epsimin, low_, upp_, alfa, beta, p0, q0, P, Q, a0_, a_, b, c_, d_);
    for (int i = 0; i < n_; ++i)
        xval[i] = result.x[static_cast<std::size_t>(i)];

    ymma_ = std::move(result.y);
    zmma_ = result.z;
    lam_ = std::move(result.lam);
    xsi_ = std::move(result.xsi);
    eta_ = std::move(result.eta);
    mu_ = std::move(result.mu);
    zet_ = result.zet;
    s_ = std::move(result.s);

    xold2_ = xold1_;
    xold1_ = std::move(xold_current);
}

#pragma once

#include <vector>

class MatlabMMA
{
  public:
    MatlabMMA(int n, int m, double a0 = 1.0, double ai = 0.0, double ci = 1000.0, double di = 0.0);

    void Update(double *xval,
                const double *df0dx,
                const double *fval,
                const double *dfdx,
                const double *xmin,
                const double *xmax);

    void Reset();

  private:
    int n_;
    int m_;
    int iter_;

    double a0_;
    std::vector<double> a_;
    std::vector<double> c_;
    std::vector<double> d_;

    std::vector<double> xold1_;
    std::vector<double> xold2_;
    std::vector<double> low_;
    std::vector<double> upp_;

    std::vector<double> ymma_;
    double zmma_;
    std::vector<double> lam_;
    std::vector<double> xsi_;
    std::vector<double> eta_;
    std::vector<double> mu_;
    double zet_;
    std::vector<double> s_;
};

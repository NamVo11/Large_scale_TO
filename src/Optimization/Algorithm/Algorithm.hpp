#pragma once

#include "myjson.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Algorithm {
    std::string type = "MMA";
    double move = 0.1;

    Algorithm() = default;

    explicit Algorithm(const myjson::json& j) {
        type = myjson::string_value(j, "type", type);
        move = myjson::double_value(j, "move", move);
    }

    virtual ~Algorithm() = default;
    virtual void Update(double* xval,
                        const double* df0dx,
                        const double* fval,
                        const double* dfdx,
                        const double* xmin,
                        const double* xmax) = 0;
    virtual void Reset() = 0;
};

struct MatlabMMA : Algorithm {
    MatlabMMA() = default;
    MatlabMMA(const myjson::json& j, int n, int m, double a0 = 1.0, double ai = 0.0, double ci = 1000.0, double di = 0.0)
        : Algorithm(j),n_(n),m_(m),iter_(0),a0_(a0),
          a_(static_cast<std::size_t>(m), ai),
          c_(static_cast<std::size_t>(m), ci),
          d_(static_cast<std::size_t>(m), di),
          xold1_(static_cast<std::size_t>(n), 0.0),
          xold2_(static_cast<std::size_t>(n), 0.0),
          low_(static_cast<std::size_t>(n), 0.0),
          upp_(static_cast<std::size_t>(n), 0.0),
          ymma_(static_cast<std::size_t>(m), 0.0),
          zmma_(0.0),
          lam_(static_cast<std::size_t>(m), 0.0),
          xsi_(static_cast<std::size_t>(n), 0.0),
          eta_(static_cast<std::size_t>(n), 0.0),
          mu_(static_cast<std::size_t>(m), 0.0),
          zet_(0.0),
          s_(static_cast<std::size_t>(m), 0.0) {}

    void Update(double *xval,
                const double *df0dx,
                const double *fval,
                const double *dfdx,
                const double *xmin,
                const double *xmax) override;

    void Reset() override;

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

inline std::unique_ptr<Algorithm> InitializeAlgorithm(const myjson::json& j, int n, int m) {
    const std::string type = myjson::lower_ascii(j.at("type").get<std::string>());
    if (type == "mma" || type == "matlab_mma" || type == "matlabmma") return std::make_unique<MatlabMMA>(j, n, m);
    throw std::runtime_error("Unknown Algorithm.type: " + type);
}

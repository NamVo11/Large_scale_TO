#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct DynamicLinearElastic;
struct FEM;

struct DynamicLinearElasticFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 1.0;
    double scale = 1.0;
    double value = 0.0;

    DynamicLinearElasticFunctionalBase() = default;
    explicit DynamicLinearElasticFunctionalBase(const myjson::json& j) {
        weight = myjson::double_value(j, "weight", weight);
        scale = weight / initial_value;
    }
    virtual ~DynamicLinearElasticFunctionalBase() = default;

    virtual void initialize(const FEM& fem,
                            DynamicLinearElastic& analysis,
                            const Eigen::MatrixXd& xPhys);

    virtual void calculate(const FEM& fem,
                           DynamicLinearElastic& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;

    virtual Eigen::VectorXd dfdu(const FEM& fem,
                                 DynamicLinearElastic& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int istep) = 0;

    virtual Eigen::VectorXd dfdv(const FEM& fem,
                                 DynamicLinearElastic& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int istep) = 0;

    virtual Eigen::VectorXd dfda(const FEM& fem,
                                 DynamicLinearElastic& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int istep) = 0;

    virtual Eigen::VectorXd dfdFext(const FEM& fem,
                                    DynamicLinearElastic& analysis,
                                    const Eigen::MatrixXd& xPhys,
                                    int istep) = 0;

    virtual Eigen::VectorXd dfdx(const FEM& fem,
                                 DynamicLinearElastic& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int val_id) = 0;
};

struct L2ComplianceElastic : DynamicLinearElasticFunctionalBase {
    explicit L2ComplianceElastic(const myjson::json& j) : DynamicLinearElasticFunctionalBase(j) {
        function_name = "l2_compliance_elastic";
    }

    void calculate(const FEM& fem,
                   DynamicLinearElastic& analysis,
                   const Eigen::MatrixXd& xPhys) override;

    Eigen::VectorXd dfdu(const FEM& fem,
                         DynamicLinearElastic& analysis,
                         const Eigen::MatrixXd& xPhys,
                         int istep) override;

    Eigen::VectorXd dfdFext(const FEM& fem,
                            DynamicLinearElastic& analysis,
                            const Eigen::MatrixXd& xPhys,
                            int istep) override;

    Eigen::VectorXd dfdv(const FEM&,
                         DynamicLinearElastic&,
                         const Eigen::MatrixXd&,
                         int) override {
        return Eigen::VectorXd();
    }

    Eigen::VectorXd dfda(const FEM&,
                         DynamicLinearElastic&,
                         const Eigen::MatrixXd&,
                         int) override {
        return Eigen::VectorXd();
    }

    Eigen::VectorXd dfdx(const FEM&,
                         DynamicLinearElastic&,
                         const Eigen::MatrixXd&,
                         int) override {
        return Eigen::VectorXd();
    }
};

inline std::unique_ptr<DynamicLinearElasticFunctionalBase>
InitializeDynamicLinearElasticFunctional(const myjson::json& j) {
    const std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name == "l2_compliance_elastic") {
        return std::make_unique<L2ComplianceElastic>(j);
    }
    throw std::runtime_error("Unknown DynamicLinearElastic functional: " + function_name);
}

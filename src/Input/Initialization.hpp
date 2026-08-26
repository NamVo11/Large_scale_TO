#pragma once

#include "Analysis/Analysis.hpp"
#include "Optimization/Optimization.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <string>
#include <vector>

#include "myjson.hpp"

using json = myjson::json;

struct InitializationData {
    json cfg;
    json run_json;
    json analysis_all_json;
    std::unique_ptr<FEM> fem;
    std::vector<std::shared_ptr<Interpolation>> interps;
    std::vector<std::shared_ptr<BoundaryCondition>> dirich;
    std::vector<std::shared_ptr<BoundaryCondition>> neum;
    Optimization optimization;

    explicit InitializationData(const std::string& input_json_path);

    void apply_solver_options();
    void apply_runtime_environment();
};

json read_json_file(const std::string& path);
std::string lower_ascii(std::string s);

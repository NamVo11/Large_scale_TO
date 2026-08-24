#pragma once

#include "Analysis/Analysis.hpp"
#include "Optimization/Optimization.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <string>
#include <vector>

#include "myjson.hpp"

using json = myjson::json;

json read_json_file(const std::string& path);

std::string lower_ascii(std::string s);

std::unique_ptr<FEM> InitializeFEM(const json& fem_json, const Geometry& geom);
OutputData InitializeOutput(const json& output_json,
                            const json& run_json,
                            const std::string& input_json_path);
std::vector<Region> InitializeRegions(const json& regions_json, const FEM& fem);
std::vector<std::shared_ptr<BoundaryCondition>> InitializeBoundaryConditions(const json& bcs_json, const FEM& fem);
std::unique_ptr<Analysis> InitializeAnalysis(const json& analysis_json,
                                             const json& run_json,
                                             const json& interp_json,
                                             const std::vector<Material>& materials,
                                             const std::vector<Region>& regions,
                                             const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                                             const std::vector<std::shared_ptr<BoundaryCondition>>& neum);
Optimization InitializeOptimization(const json& optimization_json,
                                    const json& analysis_all_json,
                                    const json& output_json,
                                    const json& run_json,
                                    const std::string& input_json_path,
                                    const FEM& fem,
                                    const Analysis& analysis,
                                    const std::vector<Material>& materials);
void apply_solver_options_from_json(const json& solver_json);
void apply_runtime_environment_from_json(const json& run_json);

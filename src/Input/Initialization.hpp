#pragma once

#include "MatlabLikeData.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

json read_json_file(const std::string& path);

std::string lower_ascii(std::string s);

std::unique_ptr<FEM> InitializeFEM(const json& fem_json, const Geometry& geom);
OutputData InitializeOutput(const json& output_json,
                            const json& run_json,
                            const std::string& input_json_path);
std::vector<Region> InitializeRegions(const json& regions_json, const FEM& fem);
std::vector<BoundaryCondition> InitializeBoundaryConditions(const json& bcs_json, const FEM& fem);
std::unique_ptr<DynamicLinearElastic> InitializeAnalysis(const json& analysis_json,
                                                        const json& run_json,
                                                        const json& interp_json,
                                                        const std::vector<Material>& materials,
                                                        const std::vector<Region>& regions,
                                                        const std::vector<BoundaryCondition>& dirich,
                                                        const std::vector<BoundaryCondition>& neum);
Optimization InitializeOptimization(const json& optimization_json,
                                    const json& analysis_all_json,
                                    const json& output_json,
                                    const json& run_json,
                                    const std::string& input_json_path,
                                    const FEM& fem,
                                    const DynamicLinearElastic& analysis,
                                    const std::vector<Material>& materials);
void apply_solver_options_from_json(const json& solver_json);
void apply_runtime_environment_from_json(const json& run_json);

void from_json(const json& j, Geometry& g);
void from_json(const json& j, Material& m);
void from_json(const json& j, Interp& interp);
void from_json(const json& j, Region& r);
void from_json(const json& j, BoundaryCondition& bc);
void from_json(const json& j, Filtering& f);
void from_json(const json& j, Algorithm& a);
void from_json(const json& j, ObjectiveTerm& t);
void from_json(const json& j, ObjectiveGroup& g);
void from_json(const json& j, Constraint& c);

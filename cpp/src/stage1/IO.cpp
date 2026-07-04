#include "stage1/IO.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string& s) {
    auto begin = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    if (begin >= end) return "";
    return std::string(begin, end);
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(trim(item));
    return out;
}

void readConfig(const std::string& path, CaseData& data) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open config file: " + path);

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        double d = std::stod(value);
        if (key == "E0") data.material.E0 = d;
        else if (key == "Emin") data.material.Emin = d;
        else if (key == "nu") data.material.nu = d;
        else if (key == "rho0") data.material.rho0 = d;
        else if (key == "rhomin") data.material.rhomin = d;
        else if (key == "penalE") data.material.penalE = d;
        else if (key == "penalRho") data.material.penalRho = d;
        else if (key == "dt") data.time.dt = d;
        else if (key == "nstep") data.time.nstep = static_cast<int>(d);
        else if (key == "uniform_elements") data.uniformElements = (std::abs(d) > 0.5);
    }
}

} // namespace

CaseData readCaseData(const std::string& caseDir) {
    CaseData data;
    readConfig(caseDir + "/config.txt", data);

    {
        std::ifstream in(caseDir + "/nodes.csv");
        if (!in) throw std::runtime_error("Cannot open nodes.csv");
        std::string line;
        while (std::getline(in, line)) {
            auto cols = splitCsv(line);
            if (cols.size() < 3) continue;
            data.nodes.push_back({std::stod(cols[0]), std::stod(cols[1]), std::stod(cols[2])});
        }
    }

    {
        std::ifstream in(caseDir + "/elements.csv");
        if (!in) throw std::runtime_error("Cannot open elements.csv");
        std::string line;
        while (std::getline(in, line)) {
            auto cols = splitCsv(line);
            if (cols.size() < 8) continue;
            std::array<int, 8> elem{};
            for (int i = 0; i < 8; ++i) elem[i] = std::stoi(cols[i]) - 1;
            data.elements.push_back(elem);
        }
    }

    {
        std::ifstream in(caseDir + "/fixed_dofs.csv");
        if (!in) throw std::runtime_error("Cannot open fixed_dofs.csv");
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (!line.empty()) data.fixedDofs.push_back(std::stoi(line) - 1);
        }
    }

    {
        std::ifstream in(caseDir + "/loads.csv");
        if (!in) throw std::runtime_error("Cannot open loads.csv");
        std::string line;
        while (std::getline(in, line)) {
            auto cols = splitCsv(line);
            if (cols.size() < 2) continue;
            data.loads.push_back({std::stoi(cols[0]) - 1, std::stod(cols[1])});
        }
    }

    {
        std::ifstream in(caseDir + "/design.csv");
        if (!in) throw std::runtime_error("Cannot open design.csv");
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (!line.empty()) data.design.push_back(std::stod(line));
        }
    }

    data.ndof = static_cast<int>(data.nodes.size()) * 3;
    if (data.design.size() != data.elements.size()) {
        throw std::runtime_error("design.csv length must match number of elements");
    }
    return data;
}

void ensureDirectory(const std::string& path) {
    std::filesystem::create_directories(path);
}

void writeVectorCsv(const std::string& path, const std::vector<double>& values) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write file: " + path);
    out.precision(17);
    for (double v : values) out << v << "\n";
}

void writeMatrixCsvColumns(const std::string& path, const std::vector<std::vector<double>>& columns) {
    if (columns.empty()) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write file: " + path);
    out.precision(17);
    const std::size_t nrow = columns[0].size();
    const std::size_t ncol = columns.size();
    for (std::size_t i = 0; i < nrow; ++i) {
        for (std::size_t j = 0; j < ncol; ++j) {
            if (j) out << ",";
            out << columns[j][i];
        }
        out << "\n";
    }
}

void writeSolverInfo(const std::string& path, PetscInt size, PetscInt rank,
                     PetscInt ndof, PetscInt nelem, double assemblyTime,
                     double solveTime, const std::vector<PetscInt>& iterations,
                     const std::vector<PetscReal>& residuals) {
    if (rank != 0) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write file: " + path);
    out.precision(17);
    out << "mpi_size=" << size << "\n";
    out << "ndof=" << ndof << "\n";
    out << "nelem=" << nelem << "\n";
    out << "assembly_time=" << assemblyTime << "\n";
    out << "solve_time=" << solveTime << "\n";
    out << "ksp_iterations=";
    for (auto v : iterations) out << "," << v;
    out << "\n";
    out << "ksp_residuals=";
    for (auto v : residuals) out << "," << v;
    out << "\n";
}

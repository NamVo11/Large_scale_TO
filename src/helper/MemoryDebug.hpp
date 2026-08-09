#pragma once

#include <petscsys.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace MemoryDebug {

inline bool enabled() {
    const char* value = std::getenv("STAGE2_MEMORY_DEBUG");
    return value && std::string(value) != "0";
}

inline long read_status_kb(const std::string& key) {
    std::ifstream in("/proc/self/status");
    std::string label;
    long value = 0;
    std::string unit;
    while (in >> label >> value >> unit) {
        if (label == key + ":") return value;
    }
    return 0;
}

inline long read_meminfo_kb(const std::string& key) {
    std::ifstream in("/proc/meminfo");
    std::string label;
    long value = 0;
    std::string unit;
    while (in >> label >> value >> unit) {
        if (label == key + ":") return value;
    }
    return 0;
}

inline void print(const std::string& tag) {
    if (!enabled()) return;
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;

    const long rss_kb = read_status_kb("VmRSS");
    const long hwm_kb = read_status_kb("VmHWM");
    const long avail_kb = read_meminfo_kb("MemAvailable");
    std::cout << "  memory[" << tag << "]:"
              << " rss_GB=" << static_cast<double>(rss_kb) / 1024.0 / 1024.0
              << " hwm_GB=" << static_cast<double>(hwm_kb) / 1024.0 / 1024.0
              << " mem_available_GB=" << static_cast<double>(avail_kb) / 1024.0 / 1024.0
              << "\n";
}

} // namespace MemoryDebug

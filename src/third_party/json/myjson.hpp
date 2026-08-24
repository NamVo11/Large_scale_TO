#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace myjson {

using json = nlohmann::json;

inline std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline json read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input JSON: " + path);
    json cfg;
    in >> cfg;
    return cfg;
}

inline std::string string_value(const json& j, const char* key, const std::string& fallback = "") {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<std::string>() : fallback;
}

inline double double_value(const json& j, const char* key, double fallback = 0.0) {
    if (!j.contains(key) || j.at(key).is_null()) return fallback;
    if (j.at(key).is_string()) {
        const std::string value = j.at(key).get<std::string>();
        if (value.empty()) return fallback;
        return std::stod(value);
    }
    return j.at(key).get<double>();
}

inline int int_value(const json& j, const char* key, int fallback = 0) {
    if (!j.contains(key) || j.at(key).is_null()) return fallback;
    if (j.at(key).is_string()) {
        const std::string value = j.at(key).get<std::string>();
        if (value.empty()) return fallback;
        return std::stoi(value);
    }
    return j.at(key).get<int>();
}

inline bool bool_value(const json& j, const char* key, bool fallback = false) {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<bool>() : fallback;
}

inline void require_key(const json& j, const char* key, const std::string& owner) {
    if (!j.contains(key) || j.at(key).is_null()) {
        throw std::runtime_error(owner + " requires JSON key: " + key);
    }
}

inline std::string required_string(const json& j, const char* key, const std::string& owner) {
    require_key(j, key, owner);
    return j.at(key).get<std::string>();
}

inline double required_double(const json& j, const char* key, const std::string& owner) {
    require_key(j, key, owner);
    if (j.at(key).is_string()) return std::stod(j.at(key).get<std::string>());
    return j.at(key).get<double>();
}

inline int required_int(const json& j, const char* key, const std::string& owner) {
    require_key(j, key, owner);
    if (j.at(key).is_string()) return std::stoi(j.at(key).get<std::string>());
    return j.at(key).get<int>();
}

} // namespace myjson

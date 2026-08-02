#include "ROM/Utils/rom_eigen_utils.hpp"

#include <algorithm>
#include <cctype>


std::string lower_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

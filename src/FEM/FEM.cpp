#include "FEM/FEM.hpp"
#include "FEM/Element/ElementBase.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kFEMPrecomputeThreads = 8;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) ++first;
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) --last;
    return s.substr(first, last - first);
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::string token;
    std::stringstream ss(s);
    while (std::getline(ss, token, ',')) out.push_back(trim(token));
    return out;
}

std::vector<int> parse_ints_after_equals(const std::string& text) {
    std::string s = text;
    const std::size_t eq = s.find('=');
    if (eq != std::string::npos) s = s.substr(eq + 1);
    for (char& c : s) {
        if (c == ',' || c == '\t') c = ' ';
    }
    std::stringstream ss(s);
    std::vector<int> values;
    int value = 0;
    while (ss >> value) values.push_back(value);
    return values;
}

bool try_parse_int_token(const std::string& token, int& value) {
    const std::string s = trim(token);
    if (s.empty()) return false;
    std::size_t pos = 0;
    try {
        value = std::stoi(s, &pos);
    } catch (...) {
        return false;
    }
    return pos == s.size();
}

bool is_set_card(const std::string& line) {
    return line.rfind("SET ", 0) == 0 || line.rfind("SET,", 0) == 0;
}

bool is_bulk_card_start(const std::string& line) {
    const std::string s = trim(line);
    if (s.empty()) return true;
    if (s.rfind("$", 0) == 0) return true;
    if (s.rfind("BEGIN BULK", 0) == 0 || s.rfind("ENDDATA", 0) == 0) return true;
    if (s.rfind("GRID", 0) == 0 || s.rfind("CHEXA", 0) == 0 || s.rfind("CTETRA", 0) == 0) return true;
    if (s.rfind("CPENTA", 0) == 0 || s.rfind("CQUAD4", 0) == 0 || s.rfind("CTRIA3", 0) == 0) return true;
    if (is_set_card(s)) return true;
    return false;
}

int parse_set_id_from_line(const std::string& line) {
    if (line.rfind("SET,", 0) == 0) {
        const auto tokens = split_commas(line);
        if (tokens.size() < 2) throw std::runtime_error("Invalid HyperMesh SET card: " + line);
        int set_id = 0;
        if (!try_parse_int_token(tokens[1], set_id)) throw std::runtime_error("Invalid HyperMesh SET id: " + line);
        return set_id;
    }

    std::stringstream header(line);
    std::string set_word;
    int set_id = 0;
    header >> set_word >> set_id;
    return set_id;
}

std::vector<int> parse_comma_style_set_ids(const std::string& text, int set_id) {
    std::string s = text;
    for (char& c : s) {
        if (c == ',' || c == '\t') c = ' ';
    }

    std::stringstream ss(s);
    std::string token;
    std::vector<int> values;
    while (ss >> token) {
        if (token == "SET" || token == "GRID" || token == "ELEM" || token == "LIST" || token == "+") continue;
        int value = 0;
        if (try_parse_int_token(token, value)) values.push_back(value);
    }
    if (!values.empty() && values.front() == set_id) values.erase(values.begin());
    return values;
}

std::vector<int> parse_hypermesh_set_ids(const std::string& text, int set_id, bool comma_style) {
    if (!comma_style) return parse_ints_after_equals(text);
    return parse_comma_style_set_ids(text, set_id);
}

double parse_nastran_double(const std::string& text) {
    const std::string s = trim(text);
    std::size_t pos = 0;
    const double value = std::stod(s, &pos);
    if (pos == s.size()) return value;

    static const std::regex compact_exponent(R"(^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$)");
    std::smatch m;
    if (std::regex_match(s, m, compact_exponent)) {
        return std::stod(m[1].str() + "e" + m[2].str());
    }
    throw std::runtime_error("Invalid Nastran double: " + s);
}

// Hàm này trả về chuỗi nằm giữa cặp dấu ngoặc kép (" ") đầu tiên.
std::string hmset_name_from_line(const std::string& line) {
    const std::size_t q1 = line.find('"');
    if (q1 == std::string::npos) return {};
    const std::size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return line.substr(q1 + 1, q2 - q1 - 1);
}

ElementType element_type_from_card(const std::string& card, std::size_t node_count) {
    const std::string c = lower(card);
    if (c == "chexa" && node_count == 8) return ElementType::H8;
    if (c == "ctetra" && node_count == 4) return ElementType::T4;
    if (c == "cpenta" && node_count == 6) return ElementType::P6;
    if (c == "cquad4" && node_count == 4) return ElementType::Q4;
    if (c == "ctria3" && node_count == 3) return ElementType::TRI3;
    return ElementType::Unknown;
}

} // namespace

std::string element_type_name(ElementType type) {
    switch (type) {
        case ElementType::Q4: return "Q4";
        case ElementType::TRI3: return "TRI3";
        case ElementType::H8: return "H8";
        case ElementType::T4: return "T4";
        case ElementType::P6: return "P6";
        default: return "Unknown";
    }
}

void StructuredMesh::build_from_geometry(const Geometry& geom) {
    const std::string type_name = lower(type);
    Node.clear();
    Element.clear();
    ElementObjects.clear();
    ElementTypes.clear();
    ElementDofs.clear();
    elem_center.clear();
    volume.clear();
    mesh_size.clear();
    node_ids.clear();
    element_ids.clear();
    node_id_to_local.clear();
    element_id_to_local.clear();
    hm_sets_by_id.clear();
    hm_sets_by_name.clear();
    nelx = geom.nelx;
    nely = geom.nely;
    nelz = geom.nelz;

    if (type_name == "plane_stress" || type_name == "plane_strain") {
        dim = 2;
        mesh_type = "2D_Q4";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        auto nid = [ny](int i, int j) { return i * ny + j; };

        Node.reserve(static_cast<std::size_t>(nx * ny));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Node.push_back({i * ex, j * ey, 0.0});
            }
        }

        Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        for (int i = 0; i < geom.nelx; ++i) {
            for (int j = 0; j < geom.nely; ++j) {
                std::vector<int> elem{nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)};
                Element.push_back(elem);
                ElementTypes.push_back(ElementType::Q4);
                elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, 0.0});
            }
        }
    } else if (type_name == "3d_brick") {
        dim = 3;
        mesh_type = "3D_H8";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const double ez = geom.Lz / static_cast<double>(geom.nelz);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        const int nz = geom.nelz + 1;
        auto nid = [nx, ny](int i, int j, int k) { return k * nx * ny + i * ny + j; };

        Node.reserve(static_cast<std::size_t>(nx * ny * nz));
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    Node.push_back({i * ex, j * ey, k * ez});
                }
            }
        }

        Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        for (int k = 0; k < geom.nelz; ++k) {
            for (int i = 0; i < geom.nelx; ++i) {
                for (int j = 0; j < geom.nely; ++j) {
                    Element.push_back({
                        nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                        nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)});
                    ElementTypes.push_back(ElementType::H8);
                    elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, (k + 0.5) * ez});
                }
            }
        }
    } else {
        throw std::runtime_error("StructuredMesh: unsupported fem.type");
    }

    NNode = static_cast<int>(Node.size());
    NElem = static_cast<int>(Element.size());
    ndof = NNode * dim;
    node_ids.resize(static_cast<std::size_t>(NNode));
    element_ids.resize(static_cast<std::size_t>(NElem));
    for (int n = 0; n < NNode; ++n) {
        node_ids[static_cast<std::size_t>(n)] = n + 1;
        node_id_to_local[n + 1] = n;
    }
    for (int e = 0; e < NElem; ++e) {
        element_ids[static_cast<std::size_t>(e)] = e + 1;
        element_id_to_local[e + 1] = e;
    }
    AllDofs.resize(static_cast<std::size_t>(ndof));
    std::iota(AllDofs.begin(), AllDofs.end(), 0);
}

void UnstructuredMesh::read_hypermesh_fem(const std::string& fem_file) {
    std::ifstream in(fem_file);
    if (!in) throw std::runtime_error("Cannot open HyperMesh fem file: " + fem_file);

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);

    Node.clear();
    Element.clear();
    ElementObjects.clear();
    ElementTypes.clear();
    ElementDofs.clear();
    elem_center.clear();
    volume.clear();
    mesh_size.clear();
    node_ids.clear();
    element_ids.clear();
    node_id_to_local.clear();
    element_id_to_local.clear();
    hm_sets_by_id.clear();
    hm_sets_by_name.clear();

    dim = 3;
    mesh_type = "unstructured";
    uniform_element_matrices = false;

    std::unordered_map<int, HMSet> raw_sets;

    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string s = trim(lines[i]);
        if (s.empty() || s.rfind("$$", 0) == 0) continue;

        if (s.rfind("$HMSET ", 0) == 0) {
            std::stringstream ss(s);
            std::string tag;
            HMSet set;
            ss >> tag >> set.id >> set.kind;
            set.name = hmset_name_from_line(s);
            raw_sets[set.id] = set;
            continue;
        }

        if (is_set_card(s)) {
            std::string text = s;
            const bool comma_style = s.rfind("SET,", 0) == 0;
            std::size_t j = i + 1;
            while (j < lines.size()) {
                const std::string t = trim(lines[j]);
                if (comma_style) {
                    if (is_bulk_card_start(t)) break;
                } else if (t.empty() || t.rfind("$", 0) == 0 || t.rfind("BEGIN BULK", 0) == 0) {
                    break;
                }
                text += " " + t;
                ++j;
            }

            const int set_id = parse_set_id_from_line(s);
            auto it = raw_sets.find(set_id);
            if (it != raw_sets.end()) it->second.raw_ids = parse_hypermesh_set_ids(text, set_id, comma_style);
            continue;
        }

        if (s.rfind("GRID", 0) == 0) {
            const auto tokens = split_commas(s);
            if (tokens.size() < 6) throw std::runtime_error("Invalid GRID card in " + fem_file);
            const int raw_id = std::stoi(tokens[1]);
            const int local_id = static_cast<int>(Node.size());
            node_id_to_local[raw_id] = local_id;
            node_ids.push_back(raw_id);
            Node.push_back({parse_nastran_double(tokens[3]),
                            parse_nastran_double(tokens[4]),
                            parse_nastran_double(tokens[5])});
            continue;
        }

        const bool is_element_card =
            s.rfind("CHEXA", 0) == 0 ||
            s.rfind("CTETRA", 0) == 0 ||
            s.rfind("CPENTA", 0) == 0 ||
            s.rfind("CQUAD4", 0) == 0 ||
            s.rfind("CTRIA3", 0) == 0;
        if (!is_element_card) continue;

        std::string card_text = s;
        std::size_t j = i + 1;
        while (j < lines.size()) {
            std::string t = trim(lines[j]);
            if (t.rfind("+", 0) != 0) break;
            t.erase(t.begin());
            card_text += t;
            ++j;
        }
        i = j - 1;

        auto tokens = split_commas(card_text);
        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [](const std::string& t) { return t.empty(); }),
                     tokens.end());
        if (tokens.size() < 4) throw std::runtime_error("Invalid element card in " + fem_file);

        const std::string card = tokens[0];
        const int raw_elem_id = std::stoi(tokens[1]);
        std::vector<int> raw_nodes;
        raw_nodes.reserve(tokens.size() - 3);
        for (std::size_t k = 3; k < tokens.size(); ++k) raw_nodes.push_back(std::stoi(tokens[k]));
        const ElementType etype = element_type_from_card(card, raw_nodes.size());
        if (etype == ElementType::Unknown) {
            throw std::runtime_error("Unsupported first-order element card: " + card);
        }

        std::vector<int> elem;
        elem.reserve(raw_nodes.size());
        for (int raw_node : raw_nodes) {
            auto nit = node_id_to_local.find(raw_node);
            if (nit == node_id_to_local.end()) {
                throw std::runtime_error("Element references unknown GRID id");
            }
            elem.push_back(nit->second);
        }

        const int elem_local = static_cast<int>(Element.size());
        element_id_to_local[raw_elem_id] = elem_local;
        element_ids.push_back(raw_elem_id);
        Element.push_back(std::move(elem));
        ElementTypes.push_back(etype);
    }

    NNode = static_cast<int>(Node.size());
    NElem = static_cast<int>(Element.size());
    if (NNode == 0 || NElem == 0) throw std::runtime_error("HyperMesh fem file contains no usable GRID/elements");

    bool all_h8 = true;
    bool all_q4 = true;
    bool all_2d = true;
    for (ElementType t : ElementTypes) {
        all_h8 = all_h8 && (t == ElementType::H8);
        all_q4 = all_q4 && (t == ElementType::Q4);
        all_2d = all_2d && (t == ElementType::Q4 || t == ElementType::TRI3);
    }
    if (all_h8) mesh_type = "3D_H8";
    else if (all_2d) {
        mesh_type = all_q4 ? "2D_Q4" : "2D_unstructured";
        dim = 2;
    }

    ndof = NNode * dim;
    AllDofs.resize(static_cast<std::size_t>(ndof));
    std::iota(AllDofs.begin(), AllDofs.end(), 0);

    for (auto& [set_id, set] : raw_sets) {
        set.local_ids.clear();
        const bool node_set = set.kind == 1;
        const bool elem_set = set.kind == 2;
        for (int raw_id : set.raw_ids) {
            if (node_set) {
                auto it = node_id_to_local.find(raw_id);
                if (it != node_id_to_local.end()) set.local_ids.push_back(it->second);
            } else if (elem_set) {
                auto it = element_id_to_local.find(raw_id);
                if (it != element_id_to_local.end()) set.local_ids.push_back(it->second);
            }
        }
        std::sort(set.local_ids.begin(), set.local_ids.end());
        set.local_ids.erase(std::unique(set.local_ids.begin(), set.local_ids.end()), set.local_ids.end());
        hm_sets_by_id[set.id] = set;
        if (!set.name.empty()) {
            hm_sets_by_name[set.name] = set;
            hm_sets_by_name[lower(set.name)] = set;
        }
    }
}

void FEM::build_element_objects() {
    ElementObjects.clear();
    ElementObjects.reserve(static_cast<std::size_t>(NElem));
    for (int e = 0; e < NElem; ++e) {
        ElementObjects.push_back(make_element_object(*this, e));
    }
}

void FEM::precompute_element_geometry() {
    if (static_cast<int>(ElementObjects.size()) != NElem) build_element_objects();
    elem_center.resize(static_cast<std::size_t>(NElem));
    volume.resize(static_cast<std::size_t>(NElem));
    mesh_size.resize(static_cast<std::size_t>(NElem));
    nonpositive_detJ_element_ids.clear();
    std::vector<int> nonpositive_detJ_flags(static_cast<std::size_t>(NElem), 0);
#pragma omp parallel for num_threads(kFEMPrecomputeThreads) schedule(static)
    for (int e = 0; e < NElem; ++e) {
        const auto& elem_obj = ElementObjects.at(static_cast<std::size_t>(e));
        elem_center[static_cast<std::size_t>(e)] = elem_obj->center(*this);
        const double v = elem_obj->volume(*this);
        if (!(v > 0.0)) {
            throw std::runtime_error("FEM::precompute_element_geometry: non-positive element volume at element " +
                                     std::to_string(e));
        }
        volume[static_cast<std::size_t>(e)] = v;
        mesh_size[static_cast<std::size_t>(e)] = elem_obj->mesh_size(*this);
        if (!elem_obj->detJ_is_positive) nonpositive_detJ_flags[static_cast<std::size_t>(e)] = 1;
    }

    for (int e = 0; e < NElem; ++e) {
        if (nonpositive_detJ_flags[static_cast<std::size_t>(e)] == 0) continue;
        const int raw_id = e < static_cast<int>(element_ids.size())
            ? element_ids[static_cast<std::size_t>(e)]
            : e + 1;
        nonpositive_detJ_element_ids.push_back(raw_id);
    }
    if (!nonpositive_detJ_element_ids.empty()) {
        std::cerr << "[warning] FEM mesh has " << nonpositive_detJ_element_ids.size()
                  << " element(s) with non-positive signed detJ. First raw ids:";
        const std::size_t nprint = std::min<std::size_t>(10, nonpositive_detJ_element_ids.size());
        for (std::size_t i = 0; i < nprint; ++i) std::cerr << " " << nonpositive_detJ_element_ids[i];
        if (nonpositive_detJ_element_ids.size() > nprint) std::cerr << " ...";
        std::cerr << "\n";
    }
}

void FEM::precompute_element_dofs() {
    ElementDofs.resize(static_cast<std::size_t>(NElem));
#pragma omp parallel for num_threads(kFEMPrecomputeThreads) schedule(static)
    for (int e = 0; e < NElem; ++e) {
        const auto& elem = Element.at(static_cast<std::size_t>(e));
        auto& dofs = ElementDofs[static_cast<std::size_t>(e)];
        dofs.resize(elem.size() * static_cast<std::size_t>(dim));
        for (std::size_t a = 0; a < elem.size(); ++a) {
            for (int d = 0; d < dim; ++d) {
                dofs[a * static_cast<std::size_t>(dim) + static_cast<std::size_t>(d)] =
                    static_cast<PetscInt>(dim * elem[a] + d);
            }
        }
    }
}

void FEM::debug_check_element_orientation() const {
    if (ElementObjects.empty()) {
        std::cout << "FEM orientation check: ElementObjects is empty. Call build_element_objects() first.\n";
        return;
    }

    std::cout << "FEM orientation check\n"
              << "  mesh_type = " << mesh_type << "\n"
              << "  nodes     = " << NNode << "\n"
              << "  elements  = " << NElem << "\n";

    std::vector<ElementOrientationPermutation> permutations;
    for (const auto& elem : ElementObjects) {
        permutations = elem->orientation_check_permutations();
        if (!permutations.empty()) break;
    }
    if (permutations.empty()) {
        std::cout << "  no orientation permutations are defined for this element type yet.\n";
        return;
    }

    for (const ElementOrientationPermutation& test_case : permutations) {
        int checked = 0;
        int bad = 0;
        int all_negative = 0;
        double min_det = std::numeric_limits<double>::max();
        double max_det = -std::numeric_limits<double>::max();
        std::vector<int> first_bad_raw_ids;

        std::cout << "\n" << test_case.name << "\n";
        for (const auto& elem : ElementObjects) {
            if (test_case.order.size() != elem->node_ids.size()) continue;
            std::vector<int> ordered_nodes;
            ordered_nodes.reserve(test_case.order.size());
            for (int p : test_case.order) {
                if (p < 0 || p >= static_cast<int>(elem->node_ids.size())) {
                    ordered_nodes.clear();
                    break;
                }
                ordered_nodes.push_back(elem->node_ids[static_cast<std::size_t>(p)]);
            }
            if (ordered_nodes.empty()) continue;

            const std::vector<double> detJ = elem->signed_detJ_values(*this, ordered_nodes);
            if (detJ.empty()) continue;
            ++checked;
            bool any_nonpositive = false;
            bool every_negative = true;
            for (double value : detJ) {
                min_det = std::min(min_det, value);
                max_det = std::max(max_det, value);
                any_nonpositive = any_nonpositive || !(value > 0.0);
                every_negative = every_negative && (value < 0.0);
            }
            if (any_nonpositive) {
                ++bad;
                if (first_bad_raw_ids.size() < 10) first_bad_raw_ids.push_back(elem->raw_id);
            }
            if (every_negative) ++all_negative;

            if (checked <= 3) {
                std::cout << "  sample raw_id=" << elem->raw_id << " nodes";
                for (int n : ordered_nodes) {
                    const int raw_node = n < static_cast<int>(node_ids.size()) ? node_ids[static_cast<std::size_t>(n)] : n;
                    std::cout << " " << raw_node;
                }
                std::cout << " detJ";
                for (double value : detJ) std::cout << " " << std::scientific << std::setprecision(6) << value;
                std::cout << std::defaultfloat << "\n";
            }
        }

        std::cout << "  checked elements = " << checked << "\n"
                  << "  bad elements with min(detJ)<=0 = " << bad << "\n"
                  << "  elements with all Gauss detJ<0 = " << all_negative << "\n";
        if (checked > 0) {
            std::cout << "  global min/max detJ = " << std::scientific << std::setprecision(12)
                      << min_det << " " << max_det << std::defaultfloat << "\n";
        }
        if (!first_bad_raw_ids.empty()) {
            std::cout << "  first bad raw ids:";
            for (int raw_id : first_bad_raw_ids) std::cout << " " << raw_id;
            std::cout << "\n";
        }
    }
}

using Mat3 = std::array<double, 9>;

double det3(const Mat3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

Mat3 inv3(const Mat3& a) {
    const double d = det3(a);
    if (std::abs(d) < 1e-30) throw std::runtime_error("Singular H8 Jacobian");
    return {
         (a[4] * a[8] - a[5] * a[7]) / d,
        -(a[1] * a[8] - a[2] * a[7]) / d,
         (a[1] * a[5] - a[2] * a[4]) / d,
        -(a[3] * a[8] - a[5] * a[6]) / d,
         (a[0] * a[8] - a[2] * a[6]) / d,
        -(a[0] * a[5] - a[2] * a[3]) / d,
         (a[3] * a[7] - a[4] * a[6]) / d,
        -(a[0] * a[7] - a[1] * a[6]) / d,
         (a[0] * a[4] - a[1] * a[3]) / d};
}

void add_btdb(LocalDenseMatrix& ke,
              const std::vector<double>& B,
              const std::vector<double>& D,
              int strain_dim,
              int ndof,
              double weight) {
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            double v = 0.0;
            for (int a = 0; a < strain_dim; ++a) {
                double db = 0.0;
                for (int b = 0; b < strain_dim; ++b) db += D[a * strain_dim + b] * B[b * ndof + j];
                v += B[a * ndof + i] * db;
            }
            ke(i, j) += v * weight;
        }
    }
}

ElementMatrix compute_q4(const FEM& fem, int elem_id, double nu, double thickness) {
    ElementMatrix out;
    out.ndof = 8;
    out.ke = LocalDenseMatrix::Zero(out.ndof, out.ndof);
    out.me = LocalDenseMatrix::Zero(out.ndof, out.ndof);

    const auto& elem = fem.Element.at(static_cast<std::size_t>(elem_id));
    const double c = 1.0 / (1.0 - nu * nu);
    const std::vector<double> D = {
        c, c * nu, 0.0,
        c * nu, c, 0.0,
        0.0, 0.0, c * (1.0 - nu) / 2.0};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            const double N[4] = {
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)};
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};

            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(elem[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double detJ = J[0] * J[3] - J[1] * J[2];
            if (std::abs(detJ) < 1e-30) throw std::runtime_error("Singular Q4 Jacobian");
            const double invJ[4] = {J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ};

            std::vector<double> B(3 * 8, 0.0);
            for (int a = 0; a < 4; ++a) {
                const double dx = invJ[0] * dNdxi[2 * a + 0] + invJ[1] * dNdxi[2 * a + 1];
                const double dy = invJ[2] * dNdxi[2 * a + 0] + invJ[3] * dNdxi[2 * a + 1];
                B[0 * 8 + 2 * a + 0] = dx;
                B[1 * 8 + 2 * a + 1] = dy;
                B[2 * 8 + 2 * a + 0] = dy;
                B[2 * 8 + 2 * a + 1] = dx;
            }
            add_btdb(out.ke, B, D, 3, 8, std::abs(detJ) * thickness);

            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    const double v = N[a] * N[b] * std::abs(detJ) * thickness;
                    out.me(2 * a + 0, 2 * b + 0) += v;
                    out.me(2 * a + 1, 2 * b + 1) += v;
                }
            }
        }
    }
    return out;
}

ElementMatrix compute_h8(const FEM& fem, int elem_id, double nu) {
    ElementMatrix out;
    out.ndof = 24;
    out.ke = LocalDenseMatrix::Zero(out.ndof, out.ndof);
    out.me = LocalDenseMatrix::Zero(out.ndof, out.ndof);

    const auto& elem = fem.Element.at(static_cast<std::size_t>(elem_id));
    const double c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    std::vector<double> D(6 * 6, 0.0);
    D[0 * 6 + 0] = c * (1.0 - nu);
    D[0 * 6 + 1] = c * nu;
    D[0 * 6 + 2] = c * nu;
    D[1 * 6 + 0] = c * nu;
    D[1 * 6 + 1] = c * (1.0 - nu);
    D[1 * 6 + 2] = c * nu;
    D[2 * 6 + 0] = c * nu;
    D[2 * 6 + 1] = c * nu;
    D[2 * 6 + 2] = c * (1.0 - nu);
    D[3 * 6 + 3] = c * (1.0 - 2.0 * nu) / 2.0;
    D[4 * 6 + 4] = c * (1.0 - 2.0 * nu) / 2.0;
    D[5 * 6 + 5] = c * (1.0 - 2.0 * nu) / 2.0;

    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 8> N{};
                std::array<double, 24> dNdxi{};
                for (int i = 0; i < 8; ++i) {
                    const double sx = s[i][0];
                    const double sy = s[i][1];
                    const double sz = s[i][2];
                    N[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
                }
                Mat3 J{};
                for (int a = 0; a < 8; ++a) {
                    const auto& p = fem.Node.at(static_cast<std::size_t>(elem[a]));
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        J[0 * 3 + alpha] += p[0] * dNdxi[a * 3 + alpha];
                        J[1 * 3 + alpha] += p[1] * dNdxi[a * 3 + alpha];
                        J[2 * 3 + alpha] += p[2] * dNdxi[a * 3 + alpha];
                    }
                }
                const double detJ = std::abs(det3(J));
                const Mat3 iJ = inv3(J);
                std::vector<double> B(6 * 24, 0.0);
                for (int a = 0; a < 8; ++a) {
                    double d[3]{};
                    for (int beta = 0; beta < 3; ++beta) {
                        for (int alpha = 0; alpha < 3; ++alpha) d[beta] += dNdxi[a * 3 + alpha] * iJ[alpha * 3 + beta];
                    }
                    const int col = 3 * a;
                    B[0 * 24 + col + 0] = d[0];
                    B[1 * 24 + col + 1] = d[1];
                    B[2 * 24 + col + 2] = d[2];
                    B[3 * 24 + col + 0] = d[1];
                    B[3 * 24 + col + 1] = d[0];
                    B[4 * 24 + col + 1] = d[2];
                    B[4 * 24 + col + 2] = d[1];
                    B[5 * 24 + col + 0] = d[2];
                    B[5 * 24 + col + 2] = d[0];
                }
                add_btdb(out.ke, B, D, 6, 24, detJ);
                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        const double v = N[a] * N[b] * detJ;
                        for (int d = 0; d < 3; ++d) out.me(3 * a + d, 3 * b + d) += v;
                    }
                }
            }
        }
    }
    return out;
}


ElementMatrix FEM::compute_element_matrices(int elem_id, double nu, double thickness) const {
    const FEM& fem = *this;
    ElementType type = ElementType::Unknown;
    if (elem_id >= 0 && elem_id < static_cast<int>(fem.ElementTypes.size())) {
        type = fem.ElementTypes[static_cast<std::size_t>(elem_id)];
    } else if (fem.mesh_type == "2D_Q4") {
        type = ElementType::Q4;
    } else if (fem.mesh_type == "3D_H8") {
        type = ElementType::H8;
    }

    if (type == ElementType::Q4) return compute_q4(fem, elem_id, nu, thickness);
    if (type == ElementType::H8) return compute_h8(fem, elem_id, nu);
    throw std::runtime_error("compute_element_matrices: unsupported element type " + element_type_name(type));
}



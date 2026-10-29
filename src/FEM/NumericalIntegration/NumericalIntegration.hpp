#pragma once
#include <Eigen/Dense>
#include <vector>

// Toa do và trong so của một diem tích phân (CHUA nhan Jacobian).
struct IntegrationPoint {
    Eigen::Vector3d xi;
    double weight = 0.0;
};

// Tương tự NumericalIntegration trong TOP_general_v1.
// Lop nay CHI cung cấp integration rules, không tính strain/stress/stiffness.
class NumericalIntegration {
public:
    // Mục đích: tạo Gauss rule 2D cho Quad4 (mac dinh full 2x2).
    // Trả về: danh sach (xi, weight) gom 4 diem nếu order=2.
    // Gọi bởi: Quad4::integration_rule() -> Shape::update_at_gauss_point().
    static const std::vector<IntegrationPoint>& quad4(int order = 2);

    // Mục đích: tạo tensor-product Gauss rule 3D cho Hexa8.
    // Trả về: danh sach 8 diem nếu order=2.
    // Gọi bởi: Hexa8::integration_rule().
    static const std::vector<IntegrationPoint>& hexa8(int order = 2);

    // Mục đích: quy tắc tích phân tham chiếu cho tam giac 3 nut.
    // Trả về: các diem và trong so tren tam giac tham chiếu.
    // Gọi bởi: Tria3::integration_rule().
    static const std::vector<IntegrationPoint>& tria3();

    // Mục đích: quy tắc tích phân tham chiếu cho từ dien 4 nut.
    // Trả về: các diem và trong so tren từ dien tham chiếu.
    // Gọi bởi: Tetra4::integration_rule().
    static const std::vector<IntegrationPoint>& tetra4();

    // Mục đích: quy tắc tích phân tham chiếu cho lang tru tam giac 6 nut.
    // Trả về: các diem và trong so theo quy tắc tích phân được chon.
    // Gọi bởi: Penta6::integration_rule().
    static const std::vector<IntegrationPoint>& penta6();
};

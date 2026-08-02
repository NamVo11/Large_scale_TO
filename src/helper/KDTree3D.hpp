#pragma once

#include <Eigen/Dense>

#include <memory>
#include <vector>

struct KDTree3D {
    struct Node {
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
        int id = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(const Eigen::Vector3d& point_in, int id_in)
            : point(point_in), id(id_in) {}
    };

    void insert(const Eigen::Vector3d& point, int id) {
        insert_node(root, point, id, 0);
    }

    std::vector<int> searchWithinRadius(const Eigen::Vector3d& center, double radius) const {
        std::vector<int> result;
        search_node(root.get(), center, radius, 0, result);
        return result;
    }

private:
    std::unique_ptr<Node> root;

    static void insert_node(std::unique_ptr<Node>& node,
                            const Eigen::Vector3d& point,
                            int id,
                            int depth) {
        if (!node) {
            node = std::make_unique<Node>(point, id);
            return;
        }
        const int axis = depth % 3;
        if (point(axis) < node->point(axis)) {
            insert_node(node->left, point, id, depth + 1);
        } else {
            insert_node(node->right, point, id, depth + 1);
        }
    }

    static void search_node(const Node* node,
                            const Eigen::Vector3d& center,
                            double radius,
                            int depth,
                            std::vector<int>& result) {
        if (!node) return;
        if ((node->point - center).norm() <= radius) result.push_back(node->id);

        const int axis = depth % 3;
        if (center(axis) - radius < node->point(axis)) {
            search_node(node->left.get(), center, radius, depth + 1, result);
        }
        if (center(axis) + radius >= node->point(axis)) {
            search_node(node->right.get(), center, radius, depth + 1, result);
        }
    }
};

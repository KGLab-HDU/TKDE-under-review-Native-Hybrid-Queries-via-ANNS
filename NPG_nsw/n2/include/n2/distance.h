

#pragma once

#include <eigen3/Eigen/Dense>

#include "hnsw_node.h"

namespace n2 
{
class L2Distance {
public:
    inline float operator()(const float* v1, const float* v2, size_t qty) const {
        Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> p(v1, qty, 1), q(v2, qty, 1);
        return (p - q).squaredNorm();
    }
    inline float operator()(const HnswNode* n1, const HnswNode* n2, size_t qty) const {
        return (*this)(n1->GetData(), n2->GetData(), qty);
    }
};

class AngularDistance {
public:
    inline float operator()(const float* v1, const float* v2, size_t qty) const {
        Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> p(v1, qty, 1), q(v2, qty, 1);
        return 1.0 - p.dot(q);
    }
    inline float operator()(const HnswNode* n1, const HnswNode* n2, size_t qty) const {
        return (*this)(n1->GetData(), n2->GetData(), qty);
    }
};

class DotDistance {
public:
    inline float operator()(const float* v1, const float* v2, size_t qty) const {
        Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> p(v1, qty, 1), q(v2, qty, 1);
        return -p.dot(q);
    }
    inline float operator()(const HnswNode* n1, const HnswNode* n2, size_t qty) const {
        return (*this)(n1->GetData(), n2->GetData(), qty);
    }
};

} // namespace n2

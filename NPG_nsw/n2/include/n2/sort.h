

#pragma once

#include "hnsw_node.h"

namespace n2 {

class FurtherFirst {
public:
    FurtherFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
    inline float GetDistance() const { return distance_; }
    inline HnswNode* GetNode() const { return node_; }
    bool operator< (const FurtherFirst& n) const {
        return (distance_ < n.GetDistance());
    }
private:
    HnswNode* node_;
    float distance_;
};

class CloserFirst {
public:
    CloserFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
    inline float GetDistance() const { return distance_; }
    inline HnswNode* GetNode() const { return node_; }
    bool operator< (const CloserFirst& n) const {
        return (distance_ > n.GetDistance());
    }
private:
    HnswNode* node_;
    float distance_;
};

} // namespace n2

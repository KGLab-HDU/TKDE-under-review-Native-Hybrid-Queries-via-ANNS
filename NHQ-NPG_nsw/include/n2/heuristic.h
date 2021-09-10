
#pragma once
#include <cstddef>
#include <memory>
#include <mutex>

#include "common.h"
#include "distance.h"
#include "hnsw_node.h"
#include "sort.h"
#include "min_heap.h"

namespace n2 {

class BaseNeighborSelectingPolicies {
public:
    BaseNeighborSelectingPolicies() {}
    virtual ~BaseNeighborSelectingPolicies() = 0;
    virtual void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) = 0;
    
    int weight_build = 1;
};

class NaiveNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    NaiveNeighborSelectingPolicies() {}
    ~NaiveNeighborSelectingPolicies() override {}
    void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) override;
};

class HeuristicNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    HeuristicNeighborSelectingPolicies(): save_remains_(false) {}
    HeuristicNeighborSelectingPolicies(bool save_remain) : save_remains_(save_remain) {}
    ~HeuristicNeighborSelectingPolicies() override {}
     void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) override;
     
private:
    bool save_remains_;
};

} // namespace n2

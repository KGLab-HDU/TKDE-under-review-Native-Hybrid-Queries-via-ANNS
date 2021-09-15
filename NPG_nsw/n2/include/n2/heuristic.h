
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>

#include "distance.h"
#include "sort.h"

namespace n2 {

class BaseNeighborSelectingPolicies {
public:
    BaseNeighborSelectingPolicies() {}
    virtual ~BaseNeighborSelectingPolicies() = 0;
    
    virtual void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) = 0;
};

class NaiveNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    NaiveNeighborSelectingPolicies() {}
    ~NaiveNeighborSelectingPolicies() override {}
    void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) override;
};

template<typename DistFuncType>
class HeuristicNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    HeuristicNeighborSelectingPolicies(): save_remains_(false) {}
    HeuristicNeighborSelectingPolicies(bool save_remain) : save_remains_(save_remain) {}
    ~HeuristicNeighborSelectingPolicies() override {}
    /**
     * Returns selected neighbors to result
     * (analagous to SELECT-NEIGHBORS-HEURISTIC in Yu. A. Malkov's paper.)
     *
     * select_nn: if true, select 0.25 * m nearest neighbors to result without applying the heuristic algorithm
     */
    void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) override;
private:
    bool save_remains_;
    DistFuncType dist_func_;
};

} // namespace n2


#pragma once

#include <boost/heap/d_ary_heap.hpp>

namespace n2 {

typedef typename std::pair<int, float> IdDistancePair;
struct IdDistancePairMaxHeapComparer {
	bool operator()(const IdDistancePair& p1, const IdDistancePair& p2) const {
        return p1.second < p2.second;
    }
};
typedef typename boost::heap::d_ary_heap<float, boost::heap::arity<4>> DistanceMaxHeap;

} // namespace n2

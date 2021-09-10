

#pragma once

namespace n2 {

enum class GraphPostProcessing {
    SKIP = 0,
    MERGE_LEVEL0 = 1
};

enum class NeighborSelectingPolicy {
    NAIVE = 0,
    HEURISTIC = 1,
    HEURISTIC_SAVE_REMAINS = 2,
};

enum class DistanceKind {
    UNKNOWN = -1,
    ANGULAR = 0,
    L2 = 1
};

} // namespace n2

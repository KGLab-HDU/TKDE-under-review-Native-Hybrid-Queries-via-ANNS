
#pragma once
/** @file */
namespace n2 {

/**
 * Graph merging heuristic.
 */
enum class GraphPostProcessing {
    SKIP = 0, /**< Do not merge (recommended for large-scale data (over 10M)). */
    MERGE_LEVEL0 = 1 /**< Performs an additional graph build in reverse order,
    then merges edges at level 0. So, it takes twice the build time compared to
    ``"skip"`` but shows slightly higher accuracy. (recommended for data under 10M scale). */
};

/**
 * Neighbor selecting policy.
 */
enum class NeighborSelectingPolicy {
    NAIVE = 0, /**< Select closest neighbors (not recommended). */
    HEURISTIC = 1, /**< Select neighbors using algorithm4 on HNSW paper (recommended). */
    HEURISTIC_SAVE_REMAINS = 2, /**< Experimental. */
};

enum class DistanceKind {
    UNKNOWN = -1,
    ANGULAR = 0,
    L2 = 1,
    DOT = 2
};

} // namespace n2

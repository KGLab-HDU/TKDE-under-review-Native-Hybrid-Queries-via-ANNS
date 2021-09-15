
#pragma once

#include <memory>
#include <vector>
#include <random>
#include <algorithm>

#include "common.h"
#include "hnsw_model.h"

namespace n2
{
    struct Neighbor
    {
        unsigned id;
        float distance;
        bool flag;

        Neighbor() = default;
        Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

        inline bool operator<(const Neighbor &other) const
        {
            return distance < other.distance;
        }
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
    {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance)
        {
            memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance)
        {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        //check equal ID

        while (left > 0)
        {
            if (addr[left].distance < nn.distance)
                break;
            if (addr[left].id == nn.id)
                return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
            return K + 1;
        memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N)
    {
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    class HnswSearch
    {
    public:
        static std::unique_ptr<HnswSearch> GenerateSearcher(std::shared_ptr<const HnswModel> model, size_t data_dim,
                                                            DistanceKind metric);
        virtual ~HnswSearch() {}

        virtual void SearchByVector(const std::vector<float> &qvec, size_t k, int ef_search, bool ensure_k,
                                    std::vector<int> &result) = 0;
        virtual void SearchByVector(const std::vector<float> &qvec, size_t k, int ef_search, bool ensure_k,
                                    std::vector<std::pair<int, float>> &result) = 0;
        virtual void SearchById(int id, size_t k, int ef_search, bool ensure_k,
                                std::vector<int> &result) = 0;
        virtual void SearchById(int id, size_t k, int ef_search, bool ensure_k,
                                std::vector<std::pair<int, float>> &result) = 0;

        virtual void SearchByVector(const std::vector<float> &qvec, size_t k, int ef_search,
                                    std::vector<int> &result) = 0;
        unsigned discount = 0;
    };

} // namespace n2

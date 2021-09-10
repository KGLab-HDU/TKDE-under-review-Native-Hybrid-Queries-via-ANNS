

#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <vector>

#include "../third_party/spdlog/spdlog.h"

#include "base.h"
#include "mmap.h"
#include "distance.h"
#include "sort.h"
#include "heuristic.h"
#include <boost/heap/d_ary_heap.hpp>

namespace n2
{

    class VisitedList
    {
    public:
        VisitedList(int size)
            : size_(size), mark_(1)
        {
            visited_ = new unsigned int[size_];
            memset(visited_, 0, sizeof(unsigned int) * size_);
        }

        inline unsigned int GetVisitMark() const { return mark_; }
        inline unsigned int *GetVisited() const { return visited_; }
        void Reset()
        {
            if (++mark_ == 0)
            {
                mark_ = 1;
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }
        }
        ~VisitedList()
        {
            delete[] visited_;
        }

    public:
        unsigned int *visited_;
        unsigned int size_;
        unsigned int mark_;
    };

    class Hnsw
    {
    public:
        typedef typename std::pair<int, float> IdDistancePair;
        struct IdDistancePairMinHeapComparer
        {
            bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
            {
                return p1.second > p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;

        //typedef typename std::pair<int, float> IdDistancePair;
        struct IdDistancePairMaxHeapComparer
        {
            bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
            {
                return p1.second < p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<float, boost::heap::arity<4>> DistanceMaxHeap;
        Hnsw();
        Hnsw(int dim, std::string metric = "angular");
        Hnsw(const Hnsw &other);
        Hnsw(Hnsw &other);
        Hnsw(Hnsw &&other) noexcept;
        ~Hnsw();

        Hnsw &operator=(const Hnsw &other);
        Hnsw &operator=(Hnsw &&other) noexcept;
        void SetConfigs(const std::vector<std::pair<std::string, std::string>> &configs);

        bool SaveModel(const std::string &fname) const;
        bool LoadModel(const std::string &fname, const bool use_mmap = true);
        void UnloadModel();

        void AddData(const std::vector<float> &data);

        void AddAllNodeAttributes(std::vector<std::string> attributes);

        // void AllAttributes(const std::vector<string>& attribute);

        void Fit();
        void Build(int M = -1, int M0 = -1, int ef_construction = -1, int n_threads = -1, float mult = -1, NeighborSelectingPolicy neighbor_selecting = NeighborSelectingPolicy::HEURISTIC, GraphPostProcessing graph_merging = GraphPostProcessing::SKIP, bool ensure_k = false);

        int SearchByVector_new(const std::vector<float> &qvec, std::vector<std::string> attributes, size_t k, int ef_search,
                               std::vector<std::pair<int, float>> &result);

        void SearchByVector_new_violence(const std::vector<float> &qvec, std::vector<std::string> attributes, size_t k, int ef_search,
                                         std::vector<std::pair<int, float>> &result);

        //int ReturnAlreadyId(std::vector<std::string> attributes);

        void SearchById(int id, size_t k, size_t ef_search,
                        std::vector<std::pair<int, float>> &result);

        void PrintDegreeDist() const;
        void PrintConfigs() const;

        //llw
        bool SaveAttributeTable(const std::string &fname) const;
        bool LoadAttributeTable(const std::string &fname);
        int data_num() { return data_.size(); }
        int attributes_num() { return attributes_.size(); }
        int SearchByVector_nang(const std::vector<float> &qvec, std::vector<std::string> attributes, size_t k, int ef_search,
                                std::vector<int> &result);
        int SearchByVector_new(const std::vector<float> &qvec, std::vector<char> attribute, size_t k, int ef_search, std::vector<std::pair<int, float>> &result);
        void why()
        {
            int sum = 0;
            int summ = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;
            for (int i = 0; i < num_nodes_; i++)
            {
                char *level_offset = model_level0_ + i * memory_per_node_level0_;
                char *data = level_offset + sizeof(int);
                int size = *((int *)data);
                int tsum = 0;
                for (int j = 1; j <= size; ++j)
                {
                    // 第j个近邻的id
                    int flag = 1;
                    int tnum = *((int *)(data + j * sizeof(int)));
                    summ++;
                    for (int k = 0; k < attribute_number_; k++)
                    {
                        if (*((int *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int))) != *((int *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int))))
                        {
                            flag = 0;
                            break;
                        }
                    }
                    if (flag)
                    {
                        tsum++;
                        sum++;
                    }
                }
                float percent = (float)tsum / size;
                //看看友邻
                if (percent == 0)
                {
                    std::cout << "节点" << i << " ";
                    for (int k = 0; k < attribute_number_; k++)
                    {
                        std::cout << attributes_code[k][*((int *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int)))] << " ";
                    }
                    std::cout << std::endl
                              << "友邻：";
                    for (int j = 1; j <= size; ++j)
                    {
                        // 第j个近邻的id
                        int flag = 1;
                        int tnum = *((int *)(data + j * sizeof(int)));
                        summ++;
                        for (int k = 0; k < attribute_number_; k++)
                        {
                            std::cout << attributes_code[k][*((int *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int)))] << " ";
                        }
                        std::cout << "|";
                    }
                    std::cout << std::endl;
                }
            }
        };
        void statistic()
        {
            int sum = 0;
            int summ = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;

            int size1 = 0;
            int size2 = 0;
            int size3 = 0;
            int size4 = 0;
            int size5 = 0;
            int size6 = 0;
            int size7 = 0;
            int size8 = 0;
            int m = getm();
            std::cout << m << "NN" << std::endl;

            for (int i = 0; i < num_nodes_; i++)
            {
                char *level_offset = model_level0_ + i * memory_per_node_level0_;
                char *data = level_offset + sizeof(int);
                int size = *((int *)data);
                int tsum = 0;
                for (int j = 1; j <= size; ++j)
                {
                    // 第j个近邻的id
                    int flag = 1;
                    int tnum = *((int *)(data + j * sizeof(int)));
                    summ++;
                    for (int k = 0; k < attribute_number_; k++)
                    {
                        if (*((int *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int))) != *((int *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + k * sizeof(int))))
                        {
                            flag = 0;
                            break;
                        }
                    }
                    if (flag)
                    {
                        tsum++;
                        sum++;
                    }
                }
                float percent = (float)tsum / size;
                if (percent > 0 && percent < 0.2)
                {
                    sum1++;
                }
                if (percent >= 0.2 && percent < 0.4)
                {
                    sum2++;
                }
                if (percent >= 0.4 && percent < 0.6)
                {
                    sum3++;
                }
                if (percent >= 0.6 && percent < 0.8)
                {
                    sum4++;
                }
                if (percent >= 0.8 && percent < 1)
                {
                    sum5++;
                }
                if (percent == 0)
                {
                    sum6++;
                }
                if (percent == 1)
                {
                    sum7++;
                }
                float percent2 = (float)size / m;
                if (percent2 > 0 && percent2 < 0.2)
                {
                    size1++;
                }
                if (percent2 >= 0.2 && percent2 < 0.4)
                {
                    size2++;
                }
                if (percent2 >= 0.4 && percent2 < 0.6)
                {
                    size3++;
                }
                if (percent2 >= 0.6 && percent2 < 0.8)
                {
                    size4++;
                }
                if (percent2 >= 0.8 && percent2 < 1)
                {
                    size5++;
                }
                if (percent2 == 0)
                {
                    size6++;
                }
                if (percent2 == 1)
                {
                    size7++;
                }
            }
            std::cout << "0          " << sum6 << std::endl;
            std::cout << "0.0--0.2   " << sum1 << std::endl;
            std::cout << "0.2--0.4   " << sum2 << std::endl;
            std::cout << "0.4--0.6   " << sum3 << std::endl;
            std::cout << "0.6--0.8   " << sum4 << std::endl;
            std::cout << "0.8--1.0   " << sum5 << std::endl;
            std::cout << "1          " << sum7 << std::endl;
            std::cout << sum << "|" << summ << std::endl;
            std::cout << (float)sum / summ << std::endl;
            std::cout << "0          " << size6 << std::endl;
            std::cout << "0.0--0.2   " << size1 << std::endl;
            std::cout << "0.2--0.4   " << size2 << std::endl;
            std::cout << "0.4--0.6   " << size3 << std::endl;
            std::cout << "0.6--0.8   " << size4 << std::endl;
            std::cout << "0.8--1.0   " << size5 << std::endl;
            std::cout << "1          " << size7 << std::endl;
        };
        void test()
        {
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < data_dim_; j++)
                {
                    std::cout << *((float *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + j * sizeof(float))) << " ";
                }
                for (int j = 0; j < attribute_number_; j++)
                {
                    std::cout << *((int *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + j * sizeof(int))) << " ";
                }
                std::cout << std::endl;
            }
        };
        int getm()
        {
            return MaxM_;
        }

    private:
        int DrawLevel(bool use_default_rng = false);

        void BuildGraph(bool reverse);
        void AllNodeAttributes(std::vector<std::string> attributes);
        //void AddAttributes(HnswNode* qnode);
        void Insert(HnswNode *qnode);
        void Link(HnswNode *source, HnswNode *target, bool is_naive, size_t dim);
        void SearchAtLayer(const std::vector<float> &qvec, HnswNode *enterpoint, size_t ef, std::priority_queue<FurtherFirst> &result, HnswNode *qnode);

        void SearchById_(int cur_node_id, float cur_dist, const float *query_vec,
                         size_t k, size_t ef_search,
                         std::vector<std::pair<int, float>> &result);

        bool SetValuesFromModel(char *model);
        void NormalizeVector(std::vector<float> &vec);
        //void MergeEdgesOfTwoGraphs(const std::vector<HnswNode*>& another_nodes);
        size_t GetModelConfigSize();
        void SaveModelConfig(char *model);
        template <typename T>
        char *SetValueAndIncPtr(char *ptr, const T &val)
        {
            *((T *)(ptr)) = val;
            return ptr + sizeof(T);
        }
        template <typename T>
        char *GetValueAndIncPtr(char *ptr, T &val)
        {
            val = *((T *)(ptr));
            return ptr + sizeof(T);
        }

        //llw
        std::vector<char> Attribute2int(std::vector<std::string> str);
        void MakeSearchResult(size_t k, IdDistancePairMinHeap &candidates, IdDistancePairMinHeap &visited_nodes, std::vector<int> &result);

    private:
        std::shared_ptr<spdlog::logger> logger_;
        std::unique_ptr<VisitedList> search_list_;

        const std::string n2_signature = "TOROS_N2@N9R4";
        size_t M_ = 12;
        size_t MaxM_ = 12;
        size_t MaxM0_ = 24;
        size_t efConstruction_ = 320;
        float weight_build = 1;
        float weight_search = 100;
        float levelmult_ = 1 / log(1.0 * M_);
        int num_threads_ = 1;
        bool ensure_k_ = false;
        bool is_naive_ = false;
        GraphPostProcessing post_ = GraphPostProcessing::SKIP;

        BaseDistance *dist_cls_ = nullptr;
        BaseNeighborSelectingPolicies *selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
        BaseNeighborSelectingPolicies *post_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
        std::uniform_real_distribution<double> uniform_distribution_{0.0, 1.0};
        std::default_random_engine *default_rng_ = nullptr;
        std::mt19937 rng_;

        // 当前结构中的最大层
        int maxlevel_ = 0;
        //属性个数
        int attribute_number_ = 3;
        // 查询的开始节点
        HnswNode *enterpoint_ = nullptr;
        int enterpoint_id_ = 0;
        // 所有向量的集合
        std::vector<Data> data_;
        // 所有的节点属性的集合
        std::map<int, std::vector<char>> attributes_;
        //属性每一维的id-属性
        std::vector<std::vector<std::string>> attributes_code;
        // 所有属性id的数量
        //int all_id_number_;
        //存放所有id-属性
        //std::map<int,std::vector<std::string>> id_attribute_;
        //当前插入点/查询点的所有属性
        //std::map<int,std::vector<std::string>> node_attributes_;
        // 所有节点的集合
        std::vector<HnswNode *> nodes_;
        //所有的节点数
        int num_nodes_ = 0;
        DistanceKind metric_;
        // 整个模型的起始地址
        char *model_ = nullptr;
        // 整个模型所需要的内存
        long long model_byte_size_ = 0;
        // 1层的开始地址   llw
        //char* model_higher_level_ = nullptr;
        // 0层的开始地址
        char *model_level0_ = nullptr;
        // 向量维度
        size_t data_dim_ = 0;
        // 节点向量所占的内存
        long long memory_per_data_ = 0;
        // 0层中单个节点的所有邻居所占的内存+邻居数所占的内存+offset所占的内存
        long long memory_per_link_level0_ = 0;
        // 0层中单个节点的所有邻居所占的内存+邻居数所占的内存+offset所占的内存+属性id所占的内存+该节点向量所占的内存
        long long memory_per_node_level0_ = 0;
        //long long memory_per_higher_level_ = 0;    llw
        // 除了0层,每层中每个节点所有邻居所占的内存
        //long long memory_per_node_higher_level_ = 0;    llw
        //long long higher_level_offset_ = 0;     llw
        long long level0_offset_ = 0;

        Mmap *model_mmap_ = nullptr;

        mutable std::mutex node_list_guard_;
        mutable std::mutex max_level_guard_;

        // configurations
        int rng_seed_ = 17;
        bool use_default_rng_ = false;
    };

} // namespace n2

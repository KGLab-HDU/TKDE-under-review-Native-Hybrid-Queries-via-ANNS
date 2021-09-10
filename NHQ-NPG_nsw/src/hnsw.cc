#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <xmmintrin.h>
#include <random>

#include "n2/hnsw.h"
#include "n2/hnsw_node.h"
#include "n2/distance.h"
#include "n2/min_heap.h"
#include "n2/sort.h"

#define MERGE_BUFFER_ALGO_SWITCH_THRESHOLD 100

namespace n2
{

    using std::endl;
    using std::fstream;
    using std::ifstream;
    using std::max;
    using std::min;
    using std::mutex;
    using std::ofstream;
    using std::pair;
    using std::priority_queue;
    using std::setprecision;
    using std::stof;
    using std::stoi;
    using std::string;
    using std::to_string;
    using std::unique_lock;
    using std::unordered_set;
    using std::vector;

    // 访问过的节点列表
    thread_local VisitedList *visited_list_ = nullptr;

    Hnsw::Hnsw()
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        metric_ = DistanceKind::ANGULAR;
        dist_cls_ = new AngularDistance();
    }

    Hnsw::Hnsw(int dim, string metric) : data_dim_(dim)
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        if (metric == "L2" || metric == "euclidean")
        {
            metric_ = DistanceKind::L2;
            dist_cls_ = new L2Distance();
        }
        else if (metric == "angular")
        {
            metric_ = DistanceKind::ANGULAR;
            dist_cls_ = new AngularDistance();
        }
        else
        {
            throw std::runtime_error("[Error] Invalid configuration value for DistanceMethod: " + metric);
        }
    }

    Hnsw::Hnsw(const Hnsw &other)
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        model_byte_size_ = other.model_byte_size_;
        model_ = new char[model_byte_size_];
        std::copy(other.model_, other.model_ + model_byte_size_, model_);
        SetValuesFromModel(model_);
        search_list_.reset(new VisitedList(num_nodes_));
        if (metric_ == DistanceKind::ANGULAR)
        {
            dist_cls_ = new AngularDistance();
        }
        else if (metric_ == DistanceKind::L2)
        {
            dist_cls_ = new L2Distance();
        }
    }

    Hnsw::Hnsw(Hnsw &other)
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        model_byte_size_ = other.model_byte_size_;
        model_ = new char[model_byte_size_];
        std::copy(other.model_, other.model_ + model_byte_size_, model_);
        SetValuesFromModel(model_);
        search_list_.reset(new VisitedList(num_nodes_));
        if (metric_ == DistanceKind::ANGULAR)
        {
            dist_cls_ = new AngularDistance();
        }
        else if (metric_ == DistanceKind::L2)
        {
            dist_cls_ = new L2Distance();
        }
    }

    Hnsw::Hnsw(Hnsw &&other) noexcept
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        model_byte_size_ = other.model_byte_size_;
        model_ = other.model_;
        other.model_ = nullptr;
        model_mmap_ = other.model_mmap_;
        other.model_mmap_ = nullptr;
        SetValuesFromModel(model_);
        search_list_.reset(new VisitedList(num_nodes_));
        if (metric_ == DistanceKind::ANGULAR)
        {
            dist_cls_ = new AngularDistance();
        }
        else if (metric_ == DistanceKind::L2)
        {
            dist_cls_ = new L2Distance();
        }
    }

    Hnsw &Hnsw::operator=(const Hnsw &other)
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }

        if (model_)
        {
            delete[] model_;
            model_ = nullptr;
        }

        if (dist_cls_)
        {
            delete dist_cls_;
            dist_cls_ = nullptr;
        }

        model_byte_size_ = other.model_byte_size_;
        model_ = new char[model_byte_size_];
        std::copy(other.model_, other.model_ + model_byte_size_, model_);
        SetValuesFromModel(model_);
        search_list_.reset(new VisitedList(num_nodes_));
        if (metric_ == DistanceKind::ANGULAR)
        {
            dist_cls_ = new AngularDistance();
        }
        else if (metric_ == DistanceKind::L2)
        {
            dist_cls_ = new L2Distance();
        }
        return *this;
    }

    Hnsw &Hnsw::operator=(Hnsw &&other) noexcept
    {
        logger_ = spdlog::get("n2");
        if (logger_ == nullptr)
        {
            logger_ = spdlog::stdout_logger_mt("n2");
        }
        if (model_mmap_)
        {
            delete model_mmap_;
            model_mmap_ = nullptr;
        }
        else
        {
            delete[] model_;
            model_ = nullptr;
        }

        if (dist_cls_)
        {
            delete dist_cls_;
            dist_cls_ = nullptr;
        }

        model_byte_size_ = other.model_byte_size_;
        model_ = other.model_;
        other.model_ = nullptr;
        model_mmap_ = other.model_mmap_;
        other.model_mmap_ = nullptr;
        SetValuesFromModel(model_);
        search_list_.reset(new VisitedList(num_nodes_));
        if (metric_ == DistanceKind::ANGULAR)
        {
            dist_cls_ = new AngularDistance();
        }
        else if (metric_ == DistanceKind::L2)
        {
            dist_cls_ = new L2Distance();
        }
        return *this;
    }

    Hnsw::~Hnsw()
    {
        if (model_mmap_)
        {
            delete model_mmap_;
        }
        else
        {
            if (model_)
                delete[] model_;
        }
        //删除所有节点
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            delete nodes_[i];
        }

        if (default_rng_)
        {
            delete default_rng_;
        }

        if (dist_cls_)
        {
            delete dist_cls_;
        }

        if (selecting_policy_cls_)
        {
            delete selecting_policy_cls_;
        }

        if (post_policy_cls_)
        {
            delete post_policy_cls_;
        }
    }

    //初始化各项参数
    void Hnsw::SetConfigs(const vector<pair<string, string>> &configs)
    {
        bool is_levelmult_set = false;
        for (const auto &c : configs)
        {
            if (c.first == "M")
            {
                MaxM_ = M_ = (size_t)stoi(c.second);
            }
            else if (c.first == "MaxM0")
            {
                MaxM0_ = (size_t)stoi(c.second);
            }
            else if (c.first == "efConstruction")
            {
                efConstruction_ = (size_t)stoi(c.second);
            }
            else if (c.first == "NumThread")
            {
                num_threads_ = stoi(c.second);
            }
            else if (c.first == "Mult")
            {
                levelmult_ = stof(c.second);
                is_levelmult_set = true;
                //选择返回结果的选择方式
            }
            else if (c.first == "NeighborSelecting")
            {

                if (selecting_policy_cls_)
                    delete selecting_policy_cls_;

                if (c.second == "heuristic")
                {
                    selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
                    is_naive_ = false;
                }
                else if (c.second == "heuristic_save_remains")
                {
                    selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
                    is_naive_ = false;
                }
                else if (c.second == "naive")
                {
                    selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
                    is_naive_ = true;
                }
                else
                {
                    throw std::runtime_error("[Error] Invalid configuration value for NeighborSelecting: " + c.second);
                }
            }
            else if (c.first == "GraphMerging")
            {
                if (c.second == "skip")
                {
                    post_ = GraphPostProcessing::SKIP;
                }
                else if (c.second == "merge_level0")
                {
                    post_ = GraphPostProcessing::MERGE_LEVEL0;
                }
                else
                {
                    throw std::runtime_error("[Error] Invalid configuration value for GraphMerging: " + c.second);
                }
            }
            else if (c.first == "EnsureK")
            {
                if (c.second == "true")
                {
                    ensure_k_ = true;
                }
                else
                {
                    ensure_k_ = false;
                }
            }
            else if (c.first == "weight_build")
            {
                weight_build = stof(c.second);
                selecting_policy_cls_->weight_build = weight_build;
                std::cout << "weight_build : " << weight_build << std::endl;
            }
            else if (c.first == "weight_search")
            {
                weight_search = stof(c.second);
                std::cout << "weight_search : " << weight_search << std::endl;
            }
            else
            {
                throw std::runtime_error("[Error] Invalid configuration key: " + c.first);
            }
        }
        if (!is_levelmult_set)
        {
            levelmult_ = 1 / log(1.0 * M_);
        }
    }
    //初始化随机数生成器(用于决定元素存在的最高层)
    int Hnsw::DrawLevel(bool use_default_rng)
    {
        double r = use_default_rng ? uniform_distribution_(*default_rng_) : uniform_distribution_(rng_);
        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        return (int)(-log(r) * levelmult_);
    }

    void Hnsw::Build(int M, int MaxM0, int ef_construction, int n_threads, float mult, NeighborSelectingPolicy neighbor_selecting, GraphPostProcessing graph_merging, bool ensure_k)
    {
        if (M > 0)
            MaxM_ = M_ = M;
        if (MaxM0 > 0)
            MaxM0_ = MaxM0;
        if (ef_construction > 0)
            efConstruction_ = ef_construction;
        if (n_threads > 0)
            num_threads_ = n_threads;
        levelmult_ = mult > 0 ? mult : 1 / log(1.0 * M_);

        //释放原来的选择规则
        if (selecting_policy_cls_)
            delete selecting_policy_cls_;
        //初始化选择规则
        if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC)
        {
            selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
            is_naive_ = false;
        }
        else if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS)
        {
            selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
            is_naive_ = false;
        }
        else if (neighbor_selecting == NeighborSelectingPolicy::NAIVE)
        {
            selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
            is_naive_ = true;
        }
        post_ = graph_merging;
        ensure_k_ = ensure_k;
        Fit();
    }

    void Hnsw::Fit()
    {
        if (data_.size() == 0)
            throw std::runtime_error("[Error] No data to fit. Load data first.");
        //随机数生成器（用于决定元素存在的最高层）
        // if (default_rng_ == nullptr)
        //     default_rng_ = new std::default_random_engine(100);
        rng_.seed(rng_seed_);
        BuildGraph(false);
        if (post_ == GraphPostProcessing::MERGE_LEVEL0)
        {
            vector<HnswNode *> nodes_backup;
            nodes_backup.swap(nodes_);
            BuildGraph(true);
            //MergeEdgesOfTwoGraphs(nodes_backup);
            for (size_t i = 0; i < nodes_backup.size(); ++i)
            {
                delete nodes_backup[i];
            }
            nodes_backup.clear();
        }

        // 打印节点信息
        //for (int i = 0; i < nodes_.size(); i++)
        //{
        //    std::cout << "节点" << nodes_[i]->id_ << "的向量为：";
        //    for (int j = 0; j < nodes_[i]->GetData().size(); j++)
        //    {
        //        std::cout << nodes_[i]->GetData()[j] << " ";
        //    }
        //    std::cout << endl;
        //    std::cout << "节点" << nodes_[i]->id_ << "的属性个数为：" << nodes_[i]->attributes_number_ << endl;
        //    std::cout << "节点" << nodes_[i]->id_ << "的属性类别个数为：" << nodes_[i]->attributes_id_.size() << endl;
        //    std::cout << "所有的属性为：";
        //    for (int k = 0; k < id_attribute_.size(); k++)
        //    {
        //        std::cout << k << "-";
        //        for (int j = 0; j < id_attribute_[k].size(); j++)
        //        {
        //            std::cout << id_attribute_[k][j] << ",";
        //        }
        //        std::cout << "     ";
        //    }
        //    std::cout << endl;
        //    for (int k = 0; k < nodes_[i]->attributes_id_.size(); k++)
        //    {
        //        if (nodes_[i]->GetFriends(nodes_[i]->attributes_id_[k]).size() == 0)
        //        {
        //            std::cout << "节点" << nodes_[i]->id_ << "在" << nodes_[i]->attributes_id_[k] << "层的朋友节点为0" << endl;
        //        }
        //        else
        //        {
        //            std::cout << "节点" << nodes_[i]->id_ << "属性id为" << nodes_[i]->attributes_id_[k] << "的所有朋友节点为：";
        //            for (int j = 0; j < nodes_[i]->GetFriends(nodes_[i]->attributes_id_[k]).size(); j++)
        //            {
        //                std::cout << nodes_[i]->GetFriends(nodes_[i]->attributes_id_[k])[j]->GetId() << " ";
        //            }
        //        }
        //    }
        //    std::cout << endl
        //              << endl;
        //}
        enterpoint_id_ = enterpoint_->GetId();
        // 所有的节点数
        num_nodes_ = nodes_.size();
        // 15个参数的内存大小
        long long model_config_size = GetModelConfigSize();
        // 节点向量所占的内存
        memory_per_data_ = sizeof(float) * data_dim_ + sizeof(char) * attribute_number_;
        // offset所占的内存+邻居数所占的内存+0层中单个节点的所有邻居所占的内存
        memory_per_link_level0_ = sizeof(int) * (1 + MaxM_); // 1" for saving num_links
        // offset所占的内存+邻居数所占的内存+0层中单个节点的所有邻居所占的内存+该节点向量所占的内存
        memory_per_node_level0_ = memory_per_link_level0_ + memory_per_data_;
        // 0层所占的内存
        long long level0_size = memory_per_node_level0_ * data_.size();

        // 整个模型所需要的内存
        model_byte_size_ = model_config_size + level0_size;
        // 初始化模型的内存空间
        model_ = new char[model_byte_size_];
        if (model_ == NULL)
        {
            throw std::runtime_error("[Error] Fail to allocate memory for optimised index (size: " + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");
        }
        // 从地址model_开始设置模型的内存，内存大小为model_byte_size_
        memset(model_, 0, model_byte_size_);
        // 0层的开始地址
        model_level0_ = model_ + model_config_size;

        // 地址model_开始，依次存入15个参数
        SaveModelConfig(model_);
        int higher_offset = 0;
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            nodes_[i]->CopyDataAndLevel0LinksToOptIndex(model_level0_ + i * memory_per_node_level0_, 0, MaxM_);
        }
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            delete nodes_[i];
        }
        nodes_.clear();
        data_.clear();
        attributes_.clear();
    }

    void Hnsw::BuildGraph(bool reverse)
    {
        // 初始化节点集合
        nodes_.resize(data_.size());
        //std::cout << "nodes_.size:" << nodes_.size() << endl;
        // 随机生成节点所在的成
        // int level = DrawLevel(use_default_rng_);
        //当前节点的所有属性
        //AllNodeAttributes(attributes_[0]);
        // 生成第一个节点
        HnswNode *first = new HnswNode(0, &(data_[0]), attribute_number_, attributes_[0], MaxM_);

        // 将第一个节点存入节点集合
        nodes_[0] = first;
        // 当前最大层
        // maxlevel_ = level;
        // 进点
        enterpoint_ = first;
        if (reverse)
        {
#pragma omp parallel num_threads(num_threads_)
            {
                visited_list_ = new VisitedList(data_.size());

#pragma omp for schedule(dynamic, 128)
                for (size_t i = data_.size() - 1; i >= 1; --i)
                {
                    // level = DrawLevel(use_default_rng_);
                    HnswNode *qnode = new HnswNode(i, &(data_[i]), attribute_number_, attributes_[i], MaxM_);
                    // 生成该节点的所有属性
                    nodes_[i] = qnode;
                    Insert(qnode);
                }
                delete visited_list_;
                visited_list_ = nullptr;
            }
        }
        else
        {
            // 设置代码快的并行线程
#pragma omp parallel num_threads(num_threads_)
            {
                // 初始化访问过的点集
                visited_list_ = new VisitedList(data_.size());
#pragma omp for schedule(dynamic, 128)
                for (size_t i = 1; i < data_.size(); ++i)
                {
                    //if ((i + 1) % 100 == 0)
                    //{
                    //    std::cout << "第" << i + 1 << "个元素" << endl;
                    //}
                    //if (i >= 1000000)
                    //{
                    //    std::cout << "第" << i + 1 << "个元素" << endl;
                    //}

                    // std::cout<<"第"<<i<<"个元素"<<endl;
                    // 随机生成节点的层数
                    // level = DrawLevel(use_default_rng_);
                    HnswNode *qnode = new HnswNode(i, &(data_[i]), attribute_number_, attributes_[i], MaxM_);
                    // 生成该节点的所有属性
                    // 将插入点存入节点集
                    nodes_[i] = qnode;
                    // 插入元素
                    Insert(qnode);
                }
                delete visited_list_;
                visited_list_ = nullptr;
            }
        }

        search_list_.reset(new VisitedList(data_.size()));
    }

    void Hnsw::Insert(HnswNode *qnode)
    {
        // 插入点的最高层
        // int cur_level = qnode->GetLevel();
        unique_lock<mutex> *lock = nullptr;
        // if (cur_level > maxlevel_) lock = new unique_lock<mutex>(max_level_guard_);//还不懂
        // 当前最大层数
        // int maxlevel_copy = maxlevel_;
        // 进入点
        HnswNode *enterpoint = enterpoint_;
        // 插入点的向量(用于计算向量之间的距离)
        const std::vector<float> &qvec = qnode->GetData();
        // 插入点向量的起始地址
        const float *qraw = &qvec[0];
        float PORTABLE_ALIGN32 TmpRes[8];

        _mm_prefetch(&selecting_policy_cls_, _MM_HINT_T0);

        priority_queue<FurtherFirst> temp_res;
        // 查找i层中的离插入点最进的efConstruction_个节点，结果存放在temp_res中
        SearchAtLayer(qvec, enterpoint, efConstruction_, temp_res, qnode);
        selecting_policy_cls_->Select(M_, temp_res, data_dim_, dist_cls_);
        // 添加插入点和离插入点最经的efConstruction_个节点的双向链接
        while (temp_res.size() > 0)
        {
            auto *top_node = temp_res.top().GetNode();
            temp_res.pop();
            Link(top_node, qnode, is_naive_, data_dim_);
            Link(qnode, top_node, is_naive_, data_dim_);
        }

        // if (cur_level > enterpoint_->GetLevel()) {
        //     maxlevel_ = cur_level;
        //     enterpoint_ = qnode;
        // }
        if (lock != nullptr)
            delete lock;
    }

    void Hnsw::SearchAtLayer(const std::vector<float> &qvec, HnswNode *enterpoint, size_t ef, priority_queue<FurtherFirst> &result, HnswNode *qnode)
    {
        // TODO: check Node 12bytes => 8bytes
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        float PORTABLE_ALIGN32 TmpRes[8];
        // 插入点的向量的地址(用于计算向量之间的距离)
        const float *qraw = &qvec[0];

        priority_queue<CloserFirst> candidates;
        // 进点和插入点的距离
        float d = dist_cls_->Evaluate(qraw, (float *)&(enterpoint->GetData()[0]), data_dim_, TmpRes);
        float d2 = 0;
        for (int i = 0; i < attribute_number_; i++)
        {
            if (enterpoint->attributes_[i] != qnode->attributes_[i])
                d2 += weight_build;
        }
        d += d * d2 / (weight_build * attribute_number_);
        //if (d2 == 0)
        //    d2++;
        //d = d * d2 * 2 / (d + d2);

        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list_->Reset();
        unsigned int mark = visited_list_->GetVisitMark();
        // 内存大小为data_.size()*sizeof(unsigned int)
        unsigned int *visited = visited_list_->GetVisited();
        // mark=1表示已经访问
        visited[enterpoint->GetId()] = mark;

        while (!candidates.empty())
        {
            // 候选列表中离插入节点最近的CloserFirst<节点，距离>
            const CloserFirst &cand = candidates.top();
            // 返回结果列表中离插入元素最远的节点的距离
            float lowerbound = result.top().GetDistance();
            if (cand.GetDistance() > lowerbound)
                break;
            // 候选列表中离插入节点最近的节点
            HnswNode *cand_node = cand.GetNode();
            unique_lock<mutex> lock(cand_node->access_guard_);
            // 得到候选列表中离插入节点最近的节点的所有朋友节点
            const vector<HnswNode *> &neighbors = cand_node->friends_;
            // 候选列表中踢出离插入节点最进的节点
            candidates.pop();
            for (size_t j = 0; j < neighbors.size(); ++j)
            {
                _mm_prefetch((char *)&(neighbors[j]->GetData()), _MM_HINT_T0);
            }
            // 遍历所有朋友节点
            for (size_t j = 0; j < neighbors.size(); ++j)
            {
                int fid = neighbors[j]->GetId();
                if (visited[fid] != mark)
                {
                    _mm_prefetch((char *)&(neighbors[j]->GetData()), _MM_HINT_T0);
                    visited[fid] = mark;
                    d = dist_cls_->Evaluate(qraw, (float *)&neighbors[j]->GetData()[0], data_dim_, TmpRes);
                    float d2 = 0;
                    for (int k = 0; k < attribute_number_; k++)
                    {
                        if (qnode->attributes_[k] != neighbors[j]->attributes_[k])
                            d2 += weight_build;
                    }
                    d += d * d2 / (weight_build * attribute_number_);
                    //if (d2 == 0)
                    //    d2++;
                    //d = d * d2 * 2 / (d + d2);
                    if (result.size() < ef || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbors[j], d);
                        candidates.emplace(neighbors[j], d);
                        // 当结果列表大于邻居数ef时，踢出最远的节点
                        if (result.size() > ef)
                            result.pop();
                    }
                }
            }
        }
    }

    void Hnsw::Link(HnswNode *source, HnswNode *target, bool is_naive, size_t dim)
    {
        std::unique_lock<std::mutex> lock(source->access_guard_);
        std::vector<HnswNode *> &neighbors = source->friends_;
        neighbors.push_back(target);
        // 节点source在level层的邻居数是否越界
        bool shrink = neighbors.size() > source->maxsize_;
        if (!shrink)
            return;
        float PORTABLE_ALIGN32 TmpRes[8];
        //删除邻居中距离最远的有节点
        if (is_naive)
        {
            float max = dist_cls_->Evaluate((float *)&source->GetData()[0], (float *)&neighbors[0]->GetData()[0], dim, TmpRes);
            int maxi = 0;
            // 查找节点source邻居中最远的节点
            for (size_t i = 1; i < neighbors.size(); ++i)
            {
                float curd = dist_cls_->Evaluate((float *)&source->GetData()[0], (float *)&neighbors[i]->GetData()[0], dim, TmpRes);
                if (curd > max)
                {
                    max = curd;
                    maxi = i;
                }
            }
            //删除邻居中最远的节点
            neighbors.erase(neighbors.begin() + maxi);
        }
        else
        {
            std::priority_queue<FurtherFirst> tempres;
            for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
            {
                _mm_prefetch((char *)&((*iter)->GetData()), _MM_HINT_T0);
            }

            for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
            {
                float d2 = 0;
                for (int i = 0; i < attribute_number_; i++)
                {
                    if (source->attributes_[i] != (*iter)->attributes_[i])
                        d2 += weight_build;
                }
                float d = dist_cls_->Evaluate((float *)&source->data_->GetData()[0], (float *)&(*iter)->GetData()[0], dim, TmpRes);
                d += d * d2 / (weight_build * attribute_number_);
                //if (d2 == 0)
                //    d2++;
                //d = d * d2 * 2 / (d + d2);
                tempres.emplace((*iter), d);
            }
            selecting_policy_cls_->Select(tempres.size() - 1, tempres, dim, dist_cls_);
            neighbors.clear();
            while (tempres.size())
            {
                neighbors.emplace_back(tempres.top().GetNode());
                tempres.pop();
            }
        }
    }

    bool Hnsw::SaveAttributeTable(const std::string &fname) const
    {
        ofstream b_stream(fname.c_str());
        if (b_stream)
        {
            for (int i = 0; i < attribute_number_; i++)
            {
                for (int j = 0; j < attributes_code[i].size(); j++)
                {
                    b_stream << attributes_code[i][j] << " ";
                }
                b_stream << std::endl;
            }

            return (b_stream.good());
        }
        else
        {
            throw std::runtime_error("[Error] Failed to save table to file: " + fname);
        }
        return false;
    }

    bool Hnsw::LoadAttributeTable(const std::string &fname)
    {
        ifstream in;
        in.open(fname);
        if (in.is_open())
        {
            vector<vector<string>> v(attribute_number_);
            string s;
            const string c = " ";
            int query_attributes_count = 0;
            while (getline(in, s))
            {
                //const char *a = s.c_str();
                string::size_type pos1, pos2;
                pos2 = s.find(c);
                pos1 = 0;
                while (string::npos != pos2)
                {
                    v[query_attributes_count].emplace_back(s.substr(pos1, pos2 - pos1));
                    pos1 = pos2 + c.size();
                    pos2 = s.find(c, pos1);
                }
                if (pos1 != s.length())
                    v[query_attributes_count].emplace_back(s.substr(pos1));
                query_attributes_count++;
            }
            for (int i = 0; i < attribute_number_; i++)
            {
                attributes_code.emplace_back(v[i]);
            }
            in.close();
        }
        else
        {
            throw std::runtime_error("[Error] Failed to load table to file: " + fname + " not found!");
        }

        for (int i = 0; i < attributes_code.size(); i++)
        {
            for (int j = 0; j < attributes_code[i].size(); j++)
            {
                std::cout << attributes_code[i][j] << "|";
            }
            std::cout << std::endl;
        }
        return true;
    }

    bool Hnsw::SaveModel(const string &fname) const
    {
        ofstream b_stream(fname.c_str(), fstream::out | fstream::binary);
        if (b_stream)
        {
            b_stream.write(model_, model_byte_size_);
            return (b_stream.good());
        }
        else
        {
            throw std::runtime_error("[Error] Failed to save model to file: " + fname);
        }
        return false;
    }

    bool Hnsw::LoadModel(const string &fname, const bool use_mmap)
    {
        if (!use_mmap)
        {
            ifstream in;
            in.open(fname, fstream::in | fstream::binary | fstream::ate);
            if (in.is_open())
            {
                size_t size = in.tellg();
                in.seekg(0, fstream::beg);
                model_ = new char[size];
                model_byte_size_ = size;
                in.read(model_, size);
                in.close();
            }
            else
            {
                throw std::runtime_error("[Error] Failed to load model to file: " + fname + " not found!");
            }
        }
        else
        {
            model_mmap_ = new Mmap(fname.c_str());
            model_byte_size_ = model_mmap_->GetFileSize();
            model_ = model_mmap_->GetData();
        }
        char *ptr = model_;
        ptr = GetValueAndIncPtr<size_t>(ptr, M_);
        std::cout << "M_:" << M_ << endl;
        ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
        std::cout << "MaxM_:" << MaxM_ << endl;
        ptr = GetValueAndIncPtr<size_t>(ptr, MaxM0_);
        std::cout << "MaxM0_:" << MaxM0_ << endl;
        ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
        std::cout << "efConstruction_:" << efConstruction_ << endl;
        ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
        std::cout << "levelmult_:" << levelmult_ << endl;
        ptr = GetValueAndIncPtr<int>(ptr, maxlevel_);
        std::cout << "maxlevel_:" << maxlevel_ << endl;
        ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
        std::cout << "enterpoint_id_:" << enterpoint_id_ << endl;
        ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
        std::cout << "num_nodes_:" << num_nodes_ << endl;
        ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
        // std::cout<<"metric_"<<metric_<<endl;
        size_t model_data_dim = *((size_t *)(ptr));
        if (data_dim_ > 0 && model_data_dim != data_dim_)
        {
            throw std::runtime_error("[Error] index dimension(" + to_string(data_dim_) + ") != model dimension(" + to_string(model_data_dim) + ")");
        }
        ptr = GetValueAndIncPtr<size_t>(ptr, data_dim_);
        std::cout << "data_dim_:" << data_dim_ << endl;
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
        std::cout << "memory_per_data_:" << memory_per_data_ << endl;
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
        std::cout << "memory_per_link_level0_:" << memory_per_link_level0_ << endl;
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
        std::cout << "memory_per_node_level0_:" << memory_per_node_level0_ << endl;
        ptr = GetValueAndIncPtr<long long>(ptr, level0_offset_);
        std::cout << "level0_offset_:" << level0_offset_ << endl;
        ptr = GetValueAndIncPtr<int>(ptr, attribute_number_);
        std::cout << "attribute_number_:" << attribute_number_ << endl;

        long long level0_size = memory_per_node_level0_ * num_nodes_;
        long long model_config_size = GetModelConfigSize();
        model_level0_ = model_ + model_config_size;

        search_list_.reset(new VisitedList(num_nodes_));
        if (dist_cls_)
        {
            delete dist_cls_;
        }
        switch (metric_)
        {
        case DistanceKind::ANGULAR:
            dist_cls_ = new AngularDistance();
            break;
        case DistanceKind::L2:
            dist_cls_ = new L2Distance();
            break;
        default:
            throw std::runtime_error("[Error] Unknown distance metric. ");
        }
        return true;
    }

    void Hnsw::UnloadModel()
    {
        if (model_mmap_ != nullptr)
        {
            model_mmap_->UnMap();
            delete model_mmap_;
            model_mmap_ = nullptr;
            model_ = nullptr;
            model_level0_ = nullptr;
        }

        search_list_.reset(nullptr);

        if (visited_list_ != nullptr)
        {
            delete visited_list_;
            visited_list_ = nullptr;
        }
    }

    void Hnsw::AddAllNodeAttributes(std::vector<std::string> attributes)
    {
        if (model_ != nullptr)
        {
            throw std::runtime_error("[Error] This index already has a trained model. Adding an item is not allowed.");
        }
        if (attribute_number_ != attributes.size())
        {
            attribute_number_ = attributes.size();
            std::cout << "attribute number changed to " << attribute_number_ << std::endl;
        }
        if (attributes_code.size() != attribute_number_)
        {
            attributes_code.resize(attribute_number_);
        }

        int s = attributes_.size();
        std::vector<char> tmp;

        for (int i = 0; i < attributes.size(); i++)
        {
            int flag = 1;
            for (int j = 0; j < attributes_code[i].size(); j++)
            {
                if (attributes[i] == attributes_code[i][j])
                {
                    tmp.push_back(j);
                    flag--;
                    break;
                }
            }
            if (flag)
            {
                tmp.push_back(attributes_code[i].size());
                attributes_code[i].push_back(attributes[i]);
            }
        }
        attributes_[s] = tmp;
    }

    void Hnsw::AddData(const std::vector<float> &data)
    {
        if (model_ != nullptr)
        {
            throw std::runtime_error("[Error] This index already has a trained model. Adding an item is not allowed.");
        }

        if (data.size() != data_dim_)
        {
            throw std::runtime_error("[Error] Invalid dimension data inserted: " + to_string(data.size()) + ", Predefined dimension: " + to_string(data_dim_));
        }

        // std::vector<std::string> attributes = {"浙江","北京","四川","杭州"};
        // std::vector<std::string> attribute;
        // std::vector<int> alreadyId;

        // int s = attributes_.size();
        // for(int i=0;i<attribute_number_;i++){
        //     int id;
        //     bool flog = true;
        //     while(flog){
        //         flog = false;
        //         id = rand() % attributes.size();
        //         for(int j=0;j<alreadyId.size();j++){
        //             if(id == alreadyId[j]){
        //                 flog = true;
        //                 break;
        //             }
        //         }
        //     }
        //     alreadyId.push_back(id);
        //     attributes_[s].push_back(attributes[id]);
        // }
        if (metric_ == DistanceKind::ANGULAR)
        {
            vector<float> data_copy(data);
            NormalizeVector(data_copy);
            data_.emplace_back(data_copy);
        }
        else
        {
            data_.emplace_back(data);
        }
    }

    void Hnsw::NormalizeVector(std::vector<float> &vec)
    {
        float sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
        if (sum != 0.0)
        {
            sum = 1 / sqrt(sum);
            std::transform(vec.begin(), vec.end(), vec.begin(), std::bind1st(std::multiplies<float>(), sum));
        }
    }

    void Hnsw::SearchById_(int cur_node_id, float cur_dist, const float *qraw, size_t k, size_t ef_search, vector<pair<int, float>> &result)
    {
        MinHeap<float, int> dh;
        dh.push(cur_dist, cur_node_id);
        float PORTABLE_ALIGN32 TmpRes[8];

        // 小顶堆
        typedef typename MinHeap<float, int>::Item QueueItem;
        std::queue<QueueItem> q;
        search_list_->Reset();

        unsigned int mark = search_list_->GetVisitMark();
        unsigned int *visited = search_list_->GetVisited();
        bool need_sort = false;
        if (ensure_k_)
        {
            if (!result.empty())
                need_sort = true;
            for (size_t i = 0; i < result.size(); ++i)
                visited[result[i].first] = mark;
            if (visited[cur_node_id] == mark)
                return;
        }
        visited[cur_node_id] = mark;

        std::priority_queue<pair<float, int>> visited_nodes;

        int tnum;
        float d;
        QueueItem e;
        //
        float maxKey = cur_dist;
        size_t total_size = 1;
        while (dh.size() > 0 && visited_nodes.size() < (ef_search >> 1))
        {
            e = dh.top();
            dh.pop();
            // 进点的id
            cur_node_id = e.data;

            visited_nodes.emplace(e.key, e.data);

            // 最大的距离
            float topKey = maxKey;

            // 进点的近邻的起始地址
            int *data = (int *)(model_level0_ + cur_node_id * memory_per_node_level0_ + sizeof(int));
            // 进点的邻居数
            int size = *data;
            // 将所有离查询点的距离小于topKey的点存入q
            for (int j = 1; j <= size; ++j)
            {
                // 进点的第j个近邻的id
                tnum = *(data + j);
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark)
                {
                    visited[tnum] = mark;
                    // 进点的第j个近邻离查询点的距离
                    d = dist_cls_->Evaluate(qraw, (float *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
                    if (d < topKey || total_size < ef_search)
                    {
                        q.emplace(QueueItem(d, tnum));
                        ++total_size;
                    }
                }
            }
            while (!q.empty())
            {
                dh.push(q.front().key, q.front().data);
                if (q.front().key > maxKey)
                    maxKey = q.front().key;
                q.pop();
            }
        }

        vector<pair<float, int>> res_t;
        while (dh.size() && res_t.size() < k)
        {
            res_t.emplace_back(dh.top().key, dh.top().data);
            dh.pop();
        }
        while (visited_nodes.size() > k)
            visited_nodes.pop();
        while (!visited_nodes.empty())
        {
            res_t.emplace_back(visited_nodes.top());
            visited_nodes.pop();
        }
        _mm_prefetch(&res_t[0], _MM_HINT_T0);
        std::sort(res_t.begin(), res_t.end());
        size_t sz;
        if (ensure_k_)
        {
            sz = min(k - result.size(), res_t.size());
        }
        else
        {
            sz = min(k, res_t.size());
        }
        for (size_t i = 0; i < sz; ++i)
            result.push_back(pair<int, float>(res_t[i].second, res_t[i].first));
        if (ensure_k_ && need_sort)
        {
            _mm_prefetch(&result[0], _MM_HINT_T0);
            sort(result.begin(), result.end(), [](const pair<int, float> &i, const pair<int, float> &j) -> bool
                 { return i.second < j.second; });
        }
    }

    bool Hnsw::SetValuesFromModel(char *model)
    {
        if (model)
        {
            char *ptr = model;
            ptr = GetValueAndIncPtr<size_t>(ptr, M_);
            ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
            ptr = GetValueAndIncPtr<size_t>(ptr, MaxM0_);
            ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
            ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
            ptr = GetValueAndIncPtr<int>(ptr, maxlevel_);
            ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
            ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
            ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
            ptr += sizeof(size_t);
            ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
            ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
            ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
            ptr = GetValueAndIncPtr<long long>(ptr, level0_offset_);
            ptr = GetValueAndIncPtr<int>(ptr, attribute_number_);
            //0层所有节点占的内存的大小
            long long level0_size = memory_per_node_level0_ * num_nodes_;
            //以上15个参数（起始地址为model）占的内存的大小
            long long model_config_size = GetModelConfigSize();
            //0层的起始地址
            model_level0_ = model_ + model_config_size;
            return true;
        }
        return false;
    }

    void Hnsw::MakeSearchResult(size_t k, IdDistancePairMinHeap &candidates,
                                IdDistancePairMinHeap &visited_nodes, vector<int> &result)
    {

        while (result.size() < k)
        {
            if (!candidates.empty() and !visited_nodes.empty())
            {
                const IdDistancePair &c = candidates.top();
                const IdDistancePair &v = visited_nodes.top();
                if (c.second < v.second)
                {
                    result.emplace_back(c.first);
                    candidates.pop();
                }
                else
                {
                    result.emplace_back(v.first);
                    visited_nodes.pop();
                }
            }
            else if (!candidates.empty())
            {
                const IdDistancePair &c = candidates.top();
                result.emplace_back(c.first);
                candidates.pop();
            }
            else if (!visited_nodes.empty())
            {
                const IdDistancePair &v = visited_nodes.top();
                result.emplace_back(v.first);
                visited_nodes.pop();
            }
            else
            {
                break;
            }
        }
    }
    
    int Hnsw::SearchByVector_new(const std::vector<float> &qvec, std::vector<char> attribute, size_t k, int ef_search, std::vector<std::pair<int, float>> &result)
    {
        if (model_ == nullptr)
            throw std::runtime_error("[Error] Model has not loaded!");
        // TODO: check Node 12bytes => 8bytes
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        float PORTABLE_ALIGN32 TmpRes[8];
        const float *qraw = nullptr;
        // 属性id
        if (ef_search < 0)
        {
            ef_search = 400;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        // 查询点向量
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        // 当前离查询最近的点的id
        int cur_node_id = enterpoint_id_;
        // 当前离查询最近点的距离
        float cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_level0_ + cur_node_id * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
        float d2 = 0;
        for (int i = 0; i < attribute.size(); i++)
        {
            if (attribute[i] != *((char *)(model_level0_ + cur_node_id * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + i * sizeof(char))))
                d2 += weight_search;
        }
        //ws
        cur_dist += d2;

        //auto
        //cur_dist += cur_dist * d2 / (weight_search * attribute_number_);

        //if (d2 == 0)
        //    d2++;
        //cur_dist = cur_dist * d2 * 2 / (cur_dist + d2);

        // 节点比较次数
        int nub = 1;

        // 小顶堆
        typedef typename MinHeap<float, int>::Item QueueItem;
        std::queue<QueueItem> q;
        search_list_->Reset();

        MinHeap<float, int> dh;
        dh.push(cur_dist, cur_node_id);

        unsigned int mark = search_list_->GetVisitMark();
        unsigned int *visited = search_list_->GetVisited();

        bool need_sort = false;
        if (ensure_k_)
        {
            if (!result.empty())
                need_sort = true;
            for (size_t i = 0; i < result.size(); ++i)
                visited[result[i].first] = mark;
            if (visited[cur_node_id] == mark)
                return nub;
        }
        visited[cur_node_id] = mark;

        std::priority_queue<pair<float, int>> visited_nodes;

        vector<pair<int, float>> path;
        if (ensure_k_)
            path.emplace_back(cur_node_id, cur_dist);

        float d;
        int tnum;
        size_t total_size = 1;
        float maxKey = cur_dist;
        QueueItem e;

        while (dh.size() > 0 && visited_nodes.size() < (ef_search >> 1))
        {
            // 候选列表中离插入节点最近的CloserFirst<节点，距离>
            e = dh.top();
            dh.pop();
            cur_node_id = e.data;
            visited_nodes.emplace(e.key, e.data);
            float topKey = maxKey;
            // 进点在0层的地址
            char *level_offset = model_level0_ + cur_node_id * memory_per_node_level0_;
            // 近点的i层的地址
            char *data = level_offset;
            int size = *((int *)data);
            for (int j = 1; j <= size; ++j)
            {
                // 第j个近邻的id
                tnum = *((int *)(data + j * sizeof(int)));
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark)
                {
                    visited[tnum] = mark;
                    d = (dist_cls_->Evaluate(qraw, (float *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes));
                    float d2 = 0;
                    for (int i = 0; i < attribute.size(); i++)
                    {
                        if (attribute[i] != *((char *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + i * sizeof(char))))
                            d2 += weight_search;
                    }
                    d += d2;
                    //d += d * d2 / (weight_search * attribute_number_);
                    //if (d2 == 0)
                    //    d2++;
                    //d = d * d2 * 2 / (d + d2);
                    nub++;
                    if (d < topKey || total_size < ef_search)
                    {
                        q.emplace(QueueItem(d, tnum));
                        ++total_size;
                    }
                }
            }
            while (!q.empty())
            {
                dh.push(q.front().key, q.front().data);
                if (q.front().key > maxKey)
                    maxKey = q.front().key;
                q.pop();
            }
        }

        // while(!res.empty()){
        //     const FurtherFirstNew& temp = res.top();
        //     result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
        //     res.pop();
        // }

        // for(int i=0;i<k;i++){
        //     std::cout<<"candidates.size() = "<<candidates.size() <<endl;
        //     const CloserFirstNew& temp = candidates.top();
        //     result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
        //     res.pop();
        // }

        //std::cout << "比较了： " << nub << "次" << endl;

        vector<pair<float, int>> res_t;
        while (dh.size() && res_t.size() < k)
        {
            res_t.emplace_back(dh.top().key, dh.top().data);
            dh.pop();
        }
        while (visited_nodes.size() > k)
            visited_nodes.pop();
        while (!visited_nodes.empty())
        {
            res_t.emplace_back(visited_nodes.top());
            visited_nodes.pop();
        }
        _mm_prefetch(&res_t[0], _MM_HINT_T0);
        std::sort(res_t.begin(), res_t.end());
        size_t sz;
        if (ensure_k_)
        {
            sz = min(k - result.size(), res_t.size());
        }
        else
        {
            sz = min(k, res_t.size());
        }
        for (size_t i = 0; i < sz; ++i)
            result.push_back(pair<int, float>(res_t[i].second, res_t[i].first));
        if (ensure_k_ && need_sort)
        {
            _mm_prefetch(&result[0], _MM_HINT_T0);
            sort(result.begin(), result.end(), [](const pair<int, float> &i, const pair<int, float> &j) -> bool
                 { return i.second < j.second; });
        }
        return nub;
    }

    int Hnsw::SearchByVector_new(const std::vector<float> &qvec, std::vector<std::string> attributes, size_t k, int ef_search, std::vector<std::pair<int, float>> &result)
    {

        if (model_ == nullptr)
            throw std::runtime_error("[Error] Model has not loaded!");
        // TODO: check Node 12bytes => 8bytes
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        float PORTABLE_ALIGN32 TmpRes[8];
        const float *qraw = nullptr;

        std::vector<char> attribute = Attribute2int(attributes);
        if (attribute.size() != attribute_number_)
        {
            std::cout << "错误的属性";
            return 0;
        }
        // 属性id
        if (ef_search < 0)
        {
            ef_search = 400;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        // 查询点向量
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        // 当前离查询最近的点的id
        int cur_node_id = enterpoint_id_;
        // 当前离查询最近点的距离
        float cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_level0_ + cur_node_id * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
        float d2 = 0;
        for (int i = 0; i < attribute.size(); i++)
        {
            if (attribute[i] != *((char *)(model_level0_ + cur_node_id * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + i * sizeof(char))))
                d2 += weight_search;
        }
        //ws
        cur_dist += d2;

        //auto
        //cur_dist += cur_dist * d2 / (weight_search * attribute_number_);

        //if (d2 == 0)
        //    d2++;
        //cur_dist = cur_dist * d2 * 2 / (cur_dist + d2);

        // 节点比较次数
        int nub = 1;

        // 小顶堆
        typedef typename MinHeap<float, int>::Item QueueItem;
        std::queue<QueueItem> q;
        search_list_->Reset();

        MinHeap<float, int> dh;
        dh.push(cur_dist, cur_node_id);

        unsigned int mark = search_list_->GetVisitMark();
        unsigned int *visited = search_list_->GetVisited();

        bool need_sort = false;
        if (ensure_k_)
        {
            if (!result.empty())
                need_sort = true;
            for (size_t i = 0; i < result.size(); ++i)
                visited[result[i].first] = mark;
            if (visited[cur_node_id] == mark)
                return nub;
        }
        visited[cur_node_id] = mark;

        std::priority_queue<pair<float, int>> visited_nodes;

        vector<pair<int, float>> path;
        if (ensure_k_)
            path.emplace_back(cur_node_id, cur_dist);

        float d;
        int tnum;
        size_t total_size = 1;
        float maxKey = cur_dist;
        QueueItem e;

        while (dh.size() > 0 && visited_nodes.size() < (ef_search >> 1))
        {
            // 候选列表中离插入节点最近的CloserFirst<节点，距离>
            e = dh.top();
            dh.pop();
            cur_node_id = e.data;
            visited_nodes.emplace(e.key, e.data);
            float topKey = maxKey;
            // 进点在0层的地址
            char *level_offset = model_level0_ + cur_node_id * memory_per_node_level0_;
            // 近点的i层的地址
            char *data = level_offset;
            int size = *((int *)data);
            for (int j = 1; j <= size; ++j)
            {
                // 第j个近邻的id
                tnum = *((int *)(data + j * sizeof(int)));
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark)
                {
                    visited[tnum] = mark;
                    d = (dist_cls_->Evaluate(qraw, (float *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes));
                    float d2 = 0;
                    for (int i = 0; i < attribute.size(); i++)
                    {
                        if (attribute[i] != *((char *)(model_level0_ + tnum * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + i * sizeof(char))))
                            d2 += weight_search;
                    }
                    d += d2;
                    //d += d * d2 / (weight_search * attribute_number_);
                    //if (d2 == 0)
                    //    d2++;
                    //d = d * d2 * 2 / (d + d2);
                    nub++;
                    if (d < topKey || total_size < ef_search)
                    {
                        q.emplace(QueueItem(d, tnum));
                        ++total_size;
                    }
                }
            }
            while (!q.empty())
            {
                dh.push(q.front().key, q.front().data);
                if (q.front().key > maxKey)
                    maxKey = q.front().key;
                q.pop();
            }
        }

        // while(!res.empty()){
        //     const FurtherFirstNew& temp = res.top();
        //     result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
        //     res.pop();
        // }

        // for(int i=0;i<k;i++){
        //     std::cout<<"candidates.size() = "<<candidates.size() <<endl;
        //     const CloserFirstNew& temp = candidates.top();
        //     result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
        //     res.pop();
        // }

        //std::cout << "比较了： " << nub << "次" << endl;

        vector<pair<float, int>> res_t;
        while (dh.size() && res_t.size() < k)
        {
            res_t.emplace_back(dh.top().key, dh.top().data);
            dh.pop();
        }
        while (visited_nodes.size() > k)
            visited_nodes.pop();
        while (!visited_nodes.empty())
        {
            res_t.emplace_back(visited_nodes.top());
            visited_nodes.pop();
        }
        _mm_prefetch(&res_t[0], _MM_HINT_T0);
        std::sort(res_t.begin(), res_t.end());
        size_t sz;
        if (ensure_k_)
        {
            sz = min(k - result.size(), res_t.size());
        }
        else
        {
            sz = min(k, res_t.size());
        }
        for (size_t i = 0; i < sz; ++i)
            result.push_back(pair<int, float>(res_t[i].second, res_t[i].first));
        if (ensure_k_ && need_sort)
        {
            _mm_prefetch(&result[0], _MM_HINT_T0);
            sort(result.begin(), result.end(), [](const pair<int, float> &i, const pair<int, float> &j) -> bool
                 { return i.second < j.second; });
        }
        return nub;
    }

    void Hnsw::SearchByVector_new_violence(const std::vector<float> &qvec, std::vector<std::string> attributes, size_t k, int ef_search, std::vector<std::pair<int, float>> &result)
    {
        if (model_ == nullptr)
            throw std::runtime_error("[Error] Model has not loaded!");
        // TODO: check Node 12bytes => 8bytes
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        float PORTABLE_ALIGN32 TmpRes[8];
        const float *qraw = nullptr;
        std::vector<char> attribute = Attribute2int(attributes);
        if (attribute.size() != attribute_number_)
        {
            std::cout << "错误的属性";
            return;
        }

        // priority_queue<CloserFirstNew> candidates;
        priority_queue<FurtherFirstNew> res;
        // 属性id

        if (ef_search < 0)
        {
            ef_search = 50 * k;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        // 查询点向量
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        // 当前离查询最近的点的id
        int cur_node_id = enterpoint_id_;
        // 当前离查询最近点的距离
        float cur_dist;

        float d;
        // res.emplace(cur_node_id, cur_dist);
        // candidates.emplace(cur_node_id, cur_dist);

        search_list_->Reset();

        vector<pair<int, float>> path;
        if (ensure_k_)
            path.emplace_back(cur_node_id, cur_dist);

        for (int i = 0; i < num_nodes_; i++)
        {
            int flag = true;
            char *current_node_address = model_level0_ + i * memory_per_node_level0_;
            cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
            for (int j = 0; j < attribute.size(); j++)
            {
                if (attribute[j] != *((int *)(model_level0_ + i * memory_per_node_level0_ + memory_per_link_level0_ + data_dim_ * sizeof(float) + j * sizeof(int))))
                    flag = false;
            }

            if (flag && (res.size() < k || res.top().GetDistance() > cur_dist))
            {
                res.emplace(i, cur_dist);
                if (res.size() > k)
                    res.pop();
            }
        }

        std::cout << res.size() << endl;

        while (!res.empty())
        {
            const FurtherFirstNew &temp = res.top();
            result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
            res.pop();
        }
    }

    void Hnsw::SearchById(int id, size_t k, size_t ef_search, vector<pair<int, float>> &result)
    {
        if (ef_search < 0)
        {
            ef_search = 50 * k;
        }
        SearchById_(id, 0.0, (const float *)(model_level0_ + id * memory_per_node_level0_ + memory_per_link_level0_), k, ef_search, result);
    }

    // 返回15个参数所占的内存
    size_t Hnsw::GetModelConfigSize()
    {
        size_t ret = 0;
        ret += sizeof(M_);
        ret += sizeof(MaxM_);
        ret += sizeof(MaxM0_);
        ret += sizeof(efConstruction_);
        ret += sizeof(levelmult_);
        ret += sizeof(maxlevel_);
        ret += sizeof(enterpoint_id_);
        ret += sizeof(num_nodes_);
        ret += sizeof(metric_);
        ret += sizeof(data_dim_);
        ret += sizeof(memory_per_data_);
        ret += sizeof(memory_per_link_level0_);
        ret += sizeof(memory_per_node_level0_);
        ret += sizeof(level0_offset_);
        ret += sizeof(attribute_number_);

        return ret;
    }

    void Hnsw::SaveModelConfig(char *ptr)
    {
        ptr = SetValueAndIncPtr<size_t>(ptr, M_);
        ptr = SetValueAndIncPtr<size_t>(ptr, MaxM_);
        ptr = SetValueAndIncPtr<size_t>(ptr, MaxM0_);
        ptr = SetValueAndIncPtr<size_t>(ptr, efConstruction_);
        ptr = SetValueAndIncPtr<float>(ptr, levelmult_);
        ptr = SetValueAndIncPtr<int>(ptr, maxlevel_);
        ptr = SetValueAndIncPtr<int>(ptr, enterpoint_id_);
        ptr = SetValueAndIncPtr<int>(ptr, num_nodes_);
        ptr = SetValueAndIncPtr<DistanceKind>(ptr, metric_);
        ptr = SetValueAndIncPtr<size_t>(ptr, data_dim_);
        ptr = SetValueAndIncPtr<long long>(ptr, memory_per_data_);
        ptr = SetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
        ptr = SetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
        ptr = SetValueAndIncPtr<long long>(ptr, level0_offset_);
        ptr = SetValueAndIncPtr<int>(ptr, attribute_number_);
    }

    void Hnsw::PrintConfigs() const
    {
        logger_->info("HNSW configurations & status: M({}), MaxM({}), MaxM0({}), efCon({}), levelmult({}), maxlevel({}), #nodes({}), dimension of data({}), memory per data({}), memory per link level0({}), memory per node level0({}), level0 offset({})", M_, MaxM_, MaxM0_, efConstruction_, levelmult_, maxlevel_, num_nodes_, data_dim_, memory_per_data_, memory_per_link_level0_, memory_per_node_level0_, level0_offset_);
    }

    std::vector<char> Hnsw::Attribute2int(std::vector<std::string> str)
    {
        std::vector<char> tmp;
        if (str.size() != attribute_number_)
            return tmp;
        for (int i = 0; i < str.size(); i++)
        {
            for (int j = 0; j < attributes_code[i].size(); j++)
            {
                if (str[i] == attributes_code[i][j])
                    tmp.push_back(j);
            }
        }
        return tmp;
    }

} // namespace n2

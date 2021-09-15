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

        if (selecting_policy_cls_)
            delete selecting_policy_cls_;
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

        enterpoint_id_ = enterpoint_->GetId();
        num_nodes_ = nodes_.size();
        long long model_config_size = GetModelConfigSize();
        memory_per_data_ = sizeof(float) * data_dim_ + sizeof(char) * attribute_number_;
        memory_per_link_level0_ = sizeof(int) * (1 + MaxM_); // 1" for saving num_links
        memory_per_node_level0_ = memory_per_link_level0_ + memory_per_data_;
        long long level0_size = memory_per_node_level0_ * data_.size();

        model_byte_size_ = model_config_size + level0_size;
        model_ = new char[model_byte_size_];
        if (model_ == NULL)
        {
            throw std::runtime_error("[Error] Fail to allocate memory for optimised index (size: " + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");
        }
        memset(model_, 0, model_byte_size_);
        model_level0_ = model_ + model_config_size;

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
        nodes_.resize(data_.size());
        //std::cout << "nodes_.size:" << nodes_.size() << endl;
        // int level = DrawLevel(use_default_rng_);
        //AllNodeAttributes(attributes_[0]);
        HnswNode *first = new HnswNode(0, &(data_[0]), attribute_number_, attributes_[0], MaxM_);

        nodes_[0] = first;
        // maxlevel_ = level;
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
                    nodes_[i] = qnode;
                    Insert(qnode);
                }
                delete visited_list_;
                visited_list_ = nullptr;
            }
        }
        else
        {
#pragma omp parallel num_threads(num_threads_)
            {
                visited_list_ = new VisitedList(data_.size());
#pragma omp for schedule(dynamic, 128)
                for (size_t i = 1; i < data_.size(); ++i)
                {
                    HnswNode *qnode = new HnswNode(i, &(data_[i]), attribute_number_, attributes_[i], MaxM_);
                    nodes_[i] = qnode;
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
        // int cur_level = qnode->GetLevel();
        unique_lock<mutex> *lock = nullptr;
        // if (cur_level > maxlevel_) lock = new unique_lock<mutex>(max_level_guard_);//还不懂
        // int maxlevel_copy = maxlevel_;
        HnswNode *enterpoint = enterpoint_;
        const std::vector<float> &qvec = qnode->GetData();
        const float *qraw = &qvec[0];
        float PORTABLE_ALIGN32 TmpRes[8];

        _mm_prefetch(&selecting_policy_cls_, _MM_HINT_T0);

        priority_queue<FurtherFirst> temp_res;
        SearchAtLayer(qvec, enterpoint, efConstruction_, temp_res, qnode);
        selecting_policy_cls_->Select(M_, temp_res, data_dim_, dist_cls_);
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
        const float *qraw = &qvec[0];

        priority_queue<CloserFirst> candidates;
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
        unsigned int *visited = visited_list_->GetVisited();
        visited[enterpoint->GetId()] = mark;

        while (!candidates.empty())
        {
            const CloserFirst &cand = candidates.top();
            float lowerbound = result.top().GetDistance();
            if (cand.GetDistance() > lowerbound)
                break;
            HnswNode *cand_node = cand.GetNode();
            unique_lock<mutex> lock(cand_node->access_guard_);
            const vector<HnswNode *> &neighbors = cand_node->friends_;
            candidates.pop();
            for (size_t j = 0; j < neighbors.size(); ++j)
            {
                _mm_prefetch((char *)&(neighbors[j]->GetData()), _MM_HINT_T0);
            }
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
        bool shrink = neighbors.size() > source->maxsize_;
        if (!shrink)
            return;
        float PORTABLE_ALIGN32 TmpRes[8];
        if (is_naive)
        {
            float max = dist_cls_->Evaluate((float *)&source->GetData()[0], (float *)&neighbors[0]->GetData()[0], dim, TmpRes);
            int maxi = 0;
            for (size_t i = 1; i < neighbors.size(); ++i)
            {
                float curd = dist_cls_->Evaluate((float *)&source->GetData()[0], (float *)&neighbors[i]->GetData()[0], dim, TmpRes);
                if (curd > max)
                {
                    max = curd;
                    maxi = i;
                }
            }
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
            cur_node_id = e.data;

            visited_nodes.emplace(e.key, e.data);
            
            float topKey = maxKey;

            int *data = (int *)(model_level0_ + cur_node_id * memory_per_node_level0_ + sizeof(int));
            int size = *data;
            for (int j = 1; j <= size; ++j)
            {
                tnum = *(data + j);
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark)
                {
                    visited[tnum] = mark;
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
            long long level0_size = memory_per_node_level0_ * num_nodes_;
            long long model_config_size = GetModelConfigSize();
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
        if (ef_search < 0)
        {
            ef_search = 400;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        int cur_node_id = enterpoint_id_;
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

        int nub = 1;

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
            e = dh.top();
            dh.pop();
            cur_node_id = e.data;
            visited_nodes.emplace(e.key, e.data);
            float topKey = maxKey;
            char *level_offset = model_level0_ + cur_node_id * memory_per_node_level0_;
            char *data = level_offset;
            int size = *((int *)data);
            for (int j = 1; j <= size; ++j)
            {
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
            std::cout << "wrong attributes";
            return 0;
        }
        if (ef_search < 0)
        {
            ef_search = 400;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        int cur_node_id = enterpoint_id_;
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
        int nub = 1;

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
            e = dh.top();
            dh.pop();
            cur_node_id = e.data;
            visited_nodes.emplace(e.key, e.data);
            float topKey = maxKey;
            char *level_offset = model_level0_ + cur_node_id * memory_per_node_level0_;
            char *data = level_offset;
            int size = *((int *)data);
            for (int j = 1; j <= size; ++j)
            {
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
            std::cout << "wrong attributes";
            return;
        }

        // priority_queue<CloserFirstNew> candidates;
        priority_queue<FurtherFirstNew> res;

        if (ef_search < 0)
        {
            ef_search = 50 * k;
        }
        vector<float> qvec_copy(qvec);
        if (metric_ == DistanceKind::ANGULAR)
        {
            NormalizeVector(qvec_copy);
        }
        qraw = &qvec_copy[0];
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        // int maxlevel = maxlevel_;
        int cur_node_id = enterpoint_id_;
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

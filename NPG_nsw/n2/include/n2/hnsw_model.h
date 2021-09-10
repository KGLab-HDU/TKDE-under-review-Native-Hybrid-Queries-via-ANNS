// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include "common.h"

#include "hnsw_node.h"
#include "mmap.h"

namespace n2 {

class HnswModel {
public:
    static std::shared_ptr<const HnswModel> GenerateModel(const std::vector<HnswNode*> nodes, int enterpoint_id, 
                                                          int max_m, int max_m0, DistanceKind metric, int max_level,
                                                          size_t data_dim);
    static std::shared_ptr<const HnswModel> LoadModelFromFile(const std::string& fname, const bool use_mmap=true);
    ~HnswModel();

    bool SaveModelToFile(const std::string& fname) const;

    HnswModel(const HnswModel&) = delete;
    void operator=(const HnswModel&) = delete;

    inline int GetNumNodes() const { return num_nodes_; }
    inline int GetEnterpointId() const { return enterpoint_id_; }
    inline int GetMaxLevel() const { return max_level_; }
    inline int GetDataDim() const { return data_dim_; }
    inline DistanceKind GetMetric() const { return metric_; }

    inline const float* GetData(int node_id) const { 
        return (const float*)(model_level0_node_base_offset_ + node_id * memory_per_node_level0_); 
    }
    inline const int* GetHigherLevelFriendsWithSize(int node_id, int level) const {
        int offset = *((int*)(model_level0_ + node_id * memory_per_node_level0_));
        return (const int*)(model_higher_level_ + (offset+level-1) * memory_per_node_higher_level_);
    }
    inline const int* GetLevel0FriendsWithSize(int node_id) const {
        return (const int*)(model_level0_ + node_id * memory_per_node_level0_ + sizeof(int));
    }
    inline void Graph_quality1(std::vector<std::vector<unsigned>> &data) const
        {
            int data_num = data.size();
            float sum = 0;
            for (int i = 0; i < data_num; i++)
            {
                //std::cout << i << std::endl;
                int count = 0;
                const int *friends_with_size = (const int *)(model_level0_ + i * memory_per_node_level0_ + sizeof(int));
                int size = friends_with_size[0];
                for (int j = 0; j < 10 && j < size; j++)
                {
                    for (int k = 1; k <= size; k++)
                    {
                        if (friends_with_size[k] == data[i][j + 1])
                        {
                            count++;
                            break;
                        }
                    }
                }
                sum += float(count) / (size < 10 ? float(size) : 10.0);
            }
            std::cout << "Graph Accuracy: " << sum / data_num << std::endl;
        }
private:
    HnswModel(const std::vector<HnswNode*> nodes, int enterpoint_id, int max_m, int max_m0, DistanceKind metric,
              int max_level, size_t data_dim);
    HnswModel(const std::string& fname, const bool use_mmap);

    size_t GetConfigSize();

    void SaveConfigToModel();
    void LoadConfigFromModel();

    template <typename T>
    char* SetValueAndIncPtr(char* ptr, const T& val) {
        *((T*)(ptr)) = val;
        return ptr + sizeof(T);
    }
    template <typename T>
    char* GetValueAndIncPtr(char* ptr, T& val) {
        val = *((T*)(ptr));
        return ptr + sizeof(T);
    }

public:
    int enterpoint_id_;
    int num_nodes_;
    int max_level_;
    size_t data_dim_ = 0;
    
    DistanceKind metric_;

    char* model_ = nullptr;
    uint64_t model_byte_size_;
    char* model_higher_level_ = nullptr;
    char* model_level0_ = nullptr;
    char* model_level0_node_base_offset_ = nullptr;

    uint64_t memory_per_data_;
    uint64_t memory_per_link_level0_;
    uint64_t memory_per_node_level0_;
    uint64_t memory_per_node_higher_level_;
    
    Mmap* model_mmap_ = nullptr;
};

} // namespace n2

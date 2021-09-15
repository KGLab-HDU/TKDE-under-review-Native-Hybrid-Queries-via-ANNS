
#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "data.h"

namespace n2 {

class HnswNode {
public:
    explicit HnswNode(int id, const Data* data, int level, size_t max_m, size_t max_m0);
    void CopyHigherLevelLinksToOptIndex(char* mem_offset, uint64_t memory_per_node_higher_level) const;
    void CopyDataAndLevel0LinksToOptIndex(char* mem_offset, int higher_level_offset) const;

    inline int GetId() const { return id_; }
    inline int GetLevel() const { return level_; }
    inline size_t GetMaxM() const { return max_m_; }
    inline size_t GetMaxM0() const { return max_m0_; }
    
    inline const float* GetData() const { return data_->GetRawData(); }
    
    inline std::vector<HnswNode*>& GetFriends(int level) { return friends_at_layer_[level]; }
    inline void SetFriends(int level, std::vector<HnswNode*>& new_friends) {
        friends_at_layer_[level].swap(new_friends);
    }
    inline std::mutex& GetAccessGuard() { return access_guard_; }

private:
    void CopyLinksToOptIndex(char* mem_offset, int level) const;

private:
    int id_;
    const Data* data_;
    int level_;
    size_t max_m_;
    size_t max_m0_;

    std::vector<std::vector<HnswNode*>> friends_at_layer_;
    std::mutex access_guard_;
};

} // namespace n2

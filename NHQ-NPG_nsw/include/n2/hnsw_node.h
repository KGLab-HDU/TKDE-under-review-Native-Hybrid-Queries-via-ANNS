
#pragma once

#include <vector>
#include <map>
#include <string>
#include <mutex>

#include "base.h"

namespace n2 {

class HnswNode {
public:
    explicit HnswNode(int id, const Data* data, int attributesNumber, const std::vector<char>& attr, int maxsize/*, int maxsize0*/);
    //void AddAttributesLevel(int attributes_id);
    //void CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const;
    void CopyDataAndLevel0LinksToOptIndex(char* mem_offset, int higher_level_offset, int M0) const;

    inline int GetId() const { return id_; }
    //inline std::vector<int> GetAllNodeAttributesId() const { return attributes_id_; }
    inline const std::vector<float>& GetData() const { return data_->GetData(); }
    //inline const std::vector<HnswNode*>& GetFriends(int attributeId) const { return friends_at_attribute_id_.find(attributeId)->second; }
    /*
    inline void SetFriends(int attributeId, std::vector<HnswNode*>& new_friends) {
            friends_at_attribute_id_.find(attributeId)->second.swap(new_friends);
    }*/
    float GetDistanceWithAttribute(HnswNode* node_a);
    float GetDistanceWithAttribute(std::vector<char> attribute);
private:
    void CopyLinksToOptIndex(char* mem_offset, int level) const;

public:
    int id_;
    const Data* data_;
    // int level_;
    size_t maxsize_;
    // size_t maxsize0_;
    int attributes_number_;
    //存放近邻的链表
    std::vector<HnswNode*> friends_;
    //std::map<int,std::vector<HnswNode*>> friends_at_attribute_id_;
    std::vector<char> attributes_;
    //存放所有的属性id
    //std::vector<int> attributes_id_;
    
    std::mutex access_guard_;
};

} // namespace n2

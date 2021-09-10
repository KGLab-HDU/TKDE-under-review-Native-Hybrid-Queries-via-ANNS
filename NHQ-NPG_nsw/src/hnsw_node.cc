#include <vector>

#include "n2/hnsw_node.h"
#include "string.h"

namespace n2
{

    HnswNode::HnswNode(int id, const Data *data, int attributesNumber, const std::vector<char> &attr, int maxsize /*, int maxsize0*/)
        : id_(id), data_(data), attributes_number_(attributesNumber), attributes_(attr), maxsize_(maxsize)
    {
    }
    /*
void HnswNode::AddAttributesLevel(int attributes_id){
    attributes_id_.push_back(attributes_id);
    friends_at_attribute_id_[attributes_id].reserve(maxsize_+1);
}*/

    // 预先分配0层以上每层的内存
    /*
void HnswNode::CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const {
    // 1层的起始地址
    char* mem_data = mem_offset;
    std::vector<int> hash_;
    hash_.resize(attributes_id_.size()-1);
    for(int i=1;i<attributes_id_.size();i++){
        int dd = attributes_id_[i]%(attributes_id_.size()-1);
        while(hash_[dd]!=NULL){
            dd = (dd+1)%(attributes_id_.size()-1);
        }
        hash_[dd] = attributes_id_[i];
    }
    for (int i = 0; i < hash_.size(); ++i) {
        // 分配单层的内存
        CopyLinksToOptIndex(mem_data, hash_[i]);
        // 起始地址加上本层的内存大小（(friends_at_layer_[level].size()+1)*sizeof(int)）
        mem_data += memory_per_node_higher_level;
    }
}*/
    // 预先分配0层的内存
    void HnswNode::CopyDataAndLevel0LinksToOptIndex(char *mem_offset, int higher_level_offset, int maxsize) const
    {
        // 0层的起始地址
        char *mem_data = mem_offset;
        // 分配单层的内存
        CopyLinksToOptIndex(mem_data, 0);
        mem_data += (sizeof(int) + sizeof(int) * maxsize);
        auto &data = data_->GetData();
        // 分陪存储该节点向量的内存
        // for (size_t i = 0; i < data.size(); ++i)
        // {
        //     // std::cout << "test_0_4\n";
        //     *((float *)(mem_data)) = (float)data[i];
        //     mem_data += sizeof(float);
        // }
        memcpy(mem_data, data.data(), data.size() * sizeof(float));
        mem_data += data.size() * sizeof(float);
        memcpy(mem_data, attributes_.data(), attributes_.size() * sizeof(char));
        // for (size_t i = 0; i < attributes_.size(); i++)
        // {
        //     *((char *)(mem_data)) = (char)attributes_[i];
        //     std::cout << (int)attributes_[i] << std::endl;
        //     mem_data += sizeof(char);
        // }
        // std::cout << "test_0_1\n";
    }

    void HnswNode::CopyLinksToOptIndex(char *mem_offset, int level) const
    {
        // 本层存储开始的起始地址
        char *mem_data = mem_offset;
        const auto &neighbors = friends_;
        // *((int *)(mem_data)) = (int)(neighbors.size() / 2 + 1);
        // mem_data += sizeof(int);
        // 层数（除了0层）
        /*
    if(attributes_id!=0){
        *((int*)(mem_data)) = (int)(attributes_id_.size()-1);
        mem_data += sizeof(int);
        //邻居对应的属性id
        *((int*)(mem_data)) = (int)(attributes_id);
        // 本层的起始地址加上邻居的数量所需要的内存空间（sizeof(int)）即为下一步的起始地址
        mem_data += sizeof(int);
    }*/
        // mem_data指向的地址存储内容为邻居的数量
        int k = (int)(neighbors.size());
        memcpy(mem_data, &k, sizeof(int));
        // 本层的起始地址加上邻居的数量所需要的内存空间（sizeof(int)）即为下一步的起始地址
        mem_data += sizeof(int);
        // 从地址mem_data开始将每个邻居的id（int）一次存入
        // memcpy(mem_data, neighbors.data(), sizeof(int) * k);
        for (int i = 0; i < k; ++i)
        {
            *((int *)(mem_data)) = (int)neighbors[i]->GetId();
            mem_data += sizeof(int);
        }
    }

    float HnswNode::GetDistanceWithAttribute(HnswNode *node_a)
    {
        float result = 0;
        for (int i = 0; i < attributes_number_; i++)
        {
            if (node_a->attributes_[i] != attributes_[i])
                result++;
        }
        return result;
    }

    float HnswNode::GetDistanceWithAttribute(std::vector<char> attribute)
    {
        float result = 0;
        for (int i = 0; i < attributes_number_; i++)
        {
            if (attribute[i] != attributes_[i])
                result++;
        }
        return result;
    }

} // namespace n2

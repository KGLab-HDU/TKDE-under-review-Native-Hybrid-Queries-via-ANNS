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

    /*
void HnswNode::CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const {
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
        CopyLinksToOptIndex(mem_data, hash_[i]);
        mem_data += memory_per_node_higher_level;
    }
}*/
    void HnswNode::CopyDataAndLevel0LinksToOptIndex(char *mem_offset, int higher_level_offset, int maxsize) const
    {
        char *mem_data = mem_offset;
        CopyLinksToOptIndex(mem_data, 0);
        mem_data += (sizeof(int) + sizeof(int) * maxsize);
        auto &data = data_->GetData();
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
        char *mem_data = mem_offset;
        const auto &neighbors = friends_;
        // *((int *)(mem_data)) = (int)(neighbors.size() / 2 + 1);
        // mem_data += sizeof(int);
        /*
    if(attributes_id!=0){
        *((int*)(mem_data)) = (int)(attributes_id_.size()-1);
        mem_data += sizeof(int);
        *((int*)(mem_data)) = (int)(attributes_id);
        mem_data += sizeof(int);
    }*/
        int k = (int)(neighbors.size());
        memcpy(mem_data, &k, sizeof(int));
        mem_data += sizeof(int);
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

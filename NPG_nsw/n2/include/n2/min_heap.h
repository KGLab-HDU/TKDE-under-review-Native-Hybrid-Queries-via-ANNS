

#pragma once

#include <string.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <boost/heap/d_ary_heap.hpp>

namespace n2 {

template <typename KeyType, typename DataType>
class MinHeap {
public:
    class Item {
    public:
        KeyType key;
        DataType data;
        Item() {}
        Item(const KeyType& key) :key(key) {}
        Item(const KeyType& key, const DataType& data) :key(key), data(data) {}
        bool operator<(const Item& i2) const {
            return key > i2.key;
        }
    };
   
    MinHeap() {
    }

    const KeyType top_key() {
        if (v_.size() <= 0) return 0.0;
        return v_[0].key;
    }
    
    Item top() {
        if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
        return v_[0];
    }

    void pop() {
        std::pop_heap(v_.begin(), v_.end());
        v_.pop_back();
    }
    
    void push(const KeyType& key, const DataType& data) {
        v_.emplace_back(Item(key, data));
        std::push_heap(v_.begin(), v_.end());
    }

    size_t size() {
        return v_.size();
    }

private:
    std::vector<Item> v_;
};

typedef typename std::pair<int, float> IdDistancePair;
struct IdDistancePairMinHeapComparer {
	bool operator()(const IdDistancePair& p1, const IdDistancePair& p2) const {
        return p1.second > p2.second;
    }
};
typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;

} // namespace n2

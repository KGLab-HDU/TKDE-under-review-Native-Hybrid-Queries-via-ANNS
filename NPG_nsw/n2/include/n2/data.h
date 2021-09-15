

#pragma once

#include <vector>

namespace n2 {

class Data{
public:
    Data(const std::vector<float>& data) : data_(data) {}
    inline const std::vector<float>& GetData() const { return data_; };
    inline const float* GetRawData() const { return &data_[0]; };
private:
    std::vector<float> data_;
};

} // namespace n2

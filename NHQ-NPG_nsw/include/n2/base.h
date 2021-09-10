
#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace n2 {

class Data{
public:
    Data(const std::vector<float>& vec);
    inline const std::vector<float>& GetData() const { return data_; };
private:
    std::vector<float> data_;
};

float GetTimeDiff(const std::chrono::steady_clock::time_point& begin_t,
                  const std::chrono::steady_clock::time_point& end_t);

std::string GetCurrentDateTime();

} // namespace n2

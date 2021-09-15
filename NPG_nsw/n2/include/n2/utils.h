

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace n2 {

class Utils {
public:
    static void NormalizeVector(const std::vector<float>& in, std::vector<float>& out) {
        float sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
        if (sum != 0.0) {
            sum = 1 / std::sqrt(sum);
            std::transform(in.begin(), in.end(), out.begin(), std::bind1st(std::multiplies<float>(), sum));
        }
    }
};

} // namespace n2

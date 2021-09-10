
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include "n2/base.h"


namespace n2 {

using std::string;
using std::vector;

Data::Data(const vector<float>& vec)
    :data_(vec) {
}

float GetTimeDiff(const std::chrono::steady_clock::time_point& begin_t,
                  const std::chrono::steady_clock::time_point& end_t) {
    return ((float)std::chrono::duration_cast<std::chrono::microseconds>(end_t - begin_t).count()) / 1000.0 / 1000.0;
}

string GetCurrentDateTime() {
    time_t now;
    time(&now);
    struct tm* timeinfo = localtime(&now);
    char time_string[50];
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", timeinfo);
    return string(time_string);
}

} // namespace n2

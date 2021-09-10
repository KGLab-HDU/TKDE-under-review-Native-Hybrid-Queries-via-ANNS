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

#include "n2/hnsw.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>

using namespace std;

void peak_memory_footprint()
{

    unsigned iPid = (unsigned)getpid();

    std::cout << "PID: " << iPid << std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open())
    {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while (getline(info, tmp))
    {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();
}

inline void load_data(char *filename, std::vector<std::vector<float>> &res, unsigned &num, unsigned &dim)
{ // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    float *data = nullptr;
    if (!in.is_open())
    {
        std::cout << "open file error : " << filename << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();

    res.resize(num);
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            res[i].push_back(data[i * dim + j]);
        }
    }
}

inline void load_result_data(char *filename, std::vector<std::vector<unsigned>> &res, unsigned &num, unsigned &dim)
{ // 载入ground_truth.ivecs
    unsigned *data = nullptr;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error : " << filename << filename << std::endl;
        return;
    }
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    unsigned fsize = (unsigned)ss;
    num = fsize / (dim + 1) / 4;
    data = new unsigned[num * dim];
    in.seekg(0, std::ios::beg);
    for (unsigned i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();

    res.resize(num);
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            res[i].push_back(data[i * dim + j]);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << argv[0] << " data_file save_graph max_m0 ef_construction" << std::endl;
        exit(-1);
    }
    char *data_file = argv[1];
    int m = 0;
    int max_m0 = atoi(argv[3]);
    int ef_construction = atoi(argv[4]);

    std::vector<std::vector<float>> data_load;
    unsigned points_num, dim;
    load_data(data_file, data_load, points_num, dim);

    n2::Hnsw index(dim, "L2");

    std::cout << "数据导入..." << std::endl;
    for (int i = 0; i < points_num; i++)
    {
        //std::cout << i << std::endl;
        index.AddData(data_load[i]);
    }
    std::cout << "数据导入完成" << std::endl;

    std::stringstream ss;
    std::string max_m0_str;
    ss << max_m0;
    ss >> max_m0_str;

    std::stringstream ss1;
    std::string ef_construction_str;
    ss1 << ef_construction;
    ss1 >> ef_construction_str;

    std::stringstream ss2;
    std::string m_str;
    ss2 << m;
    ss2 >> m_str;

    //int n_threads = 10;
    std::vector<std::pair<std::string, std::string>> configs = {
        {"M", m_str},
        {"MaxM0", max_m0_str},
        {"NumThread", "64"},
        {"efConstruction", ef_construction_str},
        {"NeighborSelecting", "heuristic"},
        {"GraphMerging", "skip"}};
    index.SetConfigs(configs);
    std::cout << "属性设置" << std::endl;
    peak_memory_footprint();
    auto s = std::chrono::high_resolution_clock::now();
    index.Fit();
    auto e = std::chrono::high_resolution_clock::now();
    peak_memory_footprint();
    std::chrono::duration<double> b_diff = e - s;
    std::cout << "Time cost: " << b_diff.count() << "\n";

    index.SaveModel(argv[2]);
    
    return 0;
}

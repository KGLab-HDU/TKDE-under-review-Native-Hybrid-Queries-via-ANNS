#include "n2/hnsw.h"

#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <sstream>
#include <fstream>

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

void load_data(char *filename, float *&data, unsigned &num, unsigned &dim)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
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
}

void load_result_data(char *filename, unsigned *&data, unsigned &num, unsigned &dim)
{ // 载入ground_truth.ivecs
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error : " << filename << std::endl;
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
}

void SplitString(const string &s, vector<char> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atoi(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<int> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atoi(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<float> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(atof(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atof(s.substr(pos1).c_str()));
}

void SplitString(const string &s, vector<string> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1).c_str());

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1).c_str());
}

void load_data_txt(char *filename, unsigned &num, unsigned &dim, std::vector<std::vector<string>> &data)
{
    std::string temp;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "open file error : " << filename << std::endl;
        exit(-1);
    }
    getline(file, temp);
    std::vector<int> tmp2;
    SplitString(temp, tmp2, " ");
    num = tmp2[0];
    dim = tmp2[1];
    data.resize(num);
    int groundtruth_count = 0;
    while (getline(file, temp))
    {
        SplitString(temp, data[groundtruth_count], " ");
        groundtruth_count++;
    }
    std::cout << "读入" << data.size() << "条数据" << std::endl;
    file.close();
}

inline void load_data(char *filename, std::vector<std::vector<float>> &res, unsigned &num, unsigned &dim)
{
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

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cout << argv[0] << " graph_file attributetable_file query_file groundtruth_file attributes_query_file "
                  << std::endl;
        exit(-1);
    }

    char *query_file = argv[3];
    char *groundtruth_file = argv[4];
    char *attributes_query_file = argv[5];
    std::vector<std::vector<float>> query_load;
    unsigned *ground_load = nullptr;
    vector<vector<string>> attributes_query;
    unsigned query_num, query_dim;
    unsigned ground_num, ground_dim;
    unsigned attributes_query_num, attributes_query_dim;
    load_data(query_file, query_load, query_num, query_dim);
    load_result_data(groundtruth_file, ground_load, ground_num, ground_dim);
    load_data_txt(attributes_query_file, attributes_query_num, attributes_query_dim, attributes_query);

    n2::Hnsw index;
    index.LoadModel(argv[1]);
    index.LoadAttributeTable(argv[2]);
    vector<pair<string, string>> configs = {{"weight_search", "140000"}};
    index.SetConfigs(configs);

    int search_k = 10;
    int ef_search = 250;

    vector<vector<pair<int, float>>> result(query_num);
    auto a = std::chrono::high_resolution_clock::now();
    int act = 0;
    for (int i = 0; i < result.size(); i++)
    {
        act += index.SearchByVector_new(query_load[i], attributes_query[i], search_k, ef_search, result[i]);
    }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> s_diff = b - a;
    //评估
    int cnt = 0;
    for (unsigned i = 0; i < ground_num; i++)
    {
        for (unsigned j = 0; j < search_k; j++)
        {
            unsigned k = 0;
            for (; k < search_k; k++)
            {
                if (result[i][j].first == ground_load[i * ground_dim + k])
                    break;
            }
            if (k == search_k)
                cnt++;
        }
    }
    float acc = 1 - (float)cnt / (ground_num * search_k);
    std::cerr << "Search Time: " << s_diff.count() << " " << search_k << "NN accuracy: " << acc << " Distcount: " << act << std::endl;

    peak_memory_footprint();
}

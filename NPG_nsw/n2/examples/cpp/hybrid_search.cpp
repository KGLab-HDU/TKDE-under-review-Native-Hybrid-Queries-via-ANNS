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

void SplitString(const std::string &s, std::vector<int> &v, const std::string &c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (std::string::npos != pos2)
    {
        v.push_back(atoi(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atoi(s.substr(pos1).c_str()));
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

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        std::cout << argv[0] << " graph_path data_path query_path base_att_path query_att_path groundtruth_path"
                  << std::endl;
        exit(-1);
    }
    std::vector<std::vector<float>> data_load;
    unsigned points_num, dim;
    load_data(argv[2], data_load, points_num, dim);

    char *query_file = argv[3];
    char *groundtruth_file = argv[6];
    std::vector<std::vector<float>> query_load;
    std::vector<std::vector<unsigned>> ground_load;
    unsigned query_num, query_dim;
    unsigned ground_num, ground_dim;
    load_data(query_file, query_load, query_num, query_dim);
    load_result_data(groundtruth_file, ground_load, ground_num, ground_dim);

    char *label_data_path = argv[4];
    char *label_query_path = argv[5];
    std::vector<std::vector<string>> label_data;
    std::vector<std::vector<string>> label_query;
    unsigned label_num, label_dim;
    unsigned label_query_num, label_query_dim;
    load_data_txt(label_data_path, label_num, label_dim, label_data);
    label_data.resize(points_num);
    for (size_t i = label_num; i < points_num; i++)
    {
        label_data[i] = label_data[i - label_num];
    }
    label_num = points_num;
    load_data_txt(label_query_path, label_query_num, label_query_dim, label_query);
    int attribute_number = label_data[0].size();

    n2::Hnsw index;
    index.LoadModel(argv[1]);

    //int n_threads = 10;
    std::vector<std::pair<std::string, std::string>> configs = {{"NumThread", "1"}};
    index.SetConfigs(configs);
    std::cout << "属性设置" << std::endl;

    // 测试代码
    unsigned Search_K = 10; //搜索返回查询点的最近邻个数
    unsigned candidate = 500;
    int ef_search = 100;
    std::cout << "ef_search: " << ef_search << std::endl;

    std::vector<std::vector<int>> res;
    std::vector<std::vector<int>> result(query_num);
    res.resize(query_num);
    int distcount = 0;
    auto a = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++)
    {
        distcount += index.SearchByVector(query_load[i], candidate, size_t(ef_search), res[i]);
    }
    auto b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> s_diff = b - a;

    auto aa = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++)
    {
        int j = 0;
        while (result[i].size() < Search_K && j < res[i].size())
        {
            int flag = 1;
            if (res[i][j] >= points_num)
            {
                j++;
                continue;
            }
            for (int k = 0; k < attribute_number; k++)
            {
                if (label_query[i][k] != label_data[res[i][j]][k])
                {
                    flag = 0;
                    break;
                }
            }
            if (flag)
            {
                result[i].push_back(res[i][j]);
            }
            j++;
        }
    }
    auto bb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ss_diff = bb - aa;

    int sum = 0;
    for (int i = 0; i < result.size(); i++)
    {
        sum += result[i].size();
    }
    std::cerr << "Qualified: " << sum << " Search Time: " << s_diff.count() << " Refine Time: " << ss_diff.count() << std::endl;

    int cnt = 0;
    for (unsigned i = 0; i < ground_num; i++)
    {
        for (unsigned j = 0; j < Search_K && j < result[i].size(); j++)
        {
            unsigned k = 0;
            for (; k < Search_K; k++)
            {
                if (result[i][j] == ground_load[i][k])
                {
                    cnt++;
                    break;
                }
            }
        }
    }
    float acc = (float)cnt / (ground_num * Search_K);
    std::cerr << "Total Time: " << s_diff.count() << " " << Search_K << "NN accuracy: " << acc << " Distcount: " << distcount << std::endl;
    peak_memory_footprint();
    return 0;
}

#include <chrono>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

using namespace std;

void save_result(char *filename, std::vector<std::vector<unsigned>> &results)
{
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++)
  {
    unsigned GK = (unsigned)results[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void load_result_data(char *filename, unsigned *&data, unsigned &num, unsigned &dim)
{ 
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

void SplitString(const std::string &s, std::vector<std::string> &v, const std::string &c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;

  while (std::string::npos != pos2)
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

void SplitString(const std::string &s, std::vector<char> &v, const std::string &c)
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
  std::cout << "load " << data.size() << " data" << std::endl;
  file.close();
}

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

int main(int argc, char **argv)
{
  if (argc != 7)
  {
    std::cout << argv[0] << " graph_path attributetable_path data_path query_path query_att_path groundtruth_path"
              << std::endl;
    exit(-1);
  }

  unsigned seed = 161803398;
  srand(seed);
  //std::cerr << "Using Seed " << seed << std::endl;

  char *data_path = argv[3];
  //std::cerr << "Data Path: " << data_path << std::endl;
  unsigned points_num, dim;
  float *data_load = nullptr;
  efanna2e::load_data(data_path, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);
  char *query_path = argv[4];
  //std::cerr << "Query Path: " << query_path << std::endl;
  unsigned query_num, query_dim;
  float *query_load = nullptr;
  efanna2e::load_data(query_path, query_load, query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);


  char *groundtruth_file = argv[6];
  char *attributes_query_file = argv[5];
  unsigned *ground_load = nullptr;
  vector<vector<string>> attributes_query;
  unsigned ground_num, ground_dim;
  unsigned attributes_query_num, attributes_query_dim;
  load_result_data(groundtruth_file, ground_load, ground_num, ground_dim);
  load_data_txt(attributes_query_file, attributes_query_num, attributes_query_dim, attributes_query);
  assert(dim == query_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::FAST_L2,
                             (efanna2e::Index *)(&init_index));

  char *DNG_path = argv[1];
  //std::cerr << "DNG Path: " << DNG_path << std::endl;
  index.Load(DNG_path);
  index.LoadAttributeTable(argv[2]);
  index.OptimizeGraph(data_load);

  unsigned search_k = 10;
  float weight_search = 140000;
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", 100);
  paras.Set<float>("weight_search", weight_search);

  std::vector<std::vector<unsigned>> res(query_num);
  for (unsigned i = 0; i < query_num; i++)
    res[i].resize(search_k);

  // Warm up
  for (int loop = 0; loop < 3; ++loop)
  {
    for (unsigned i = 0; i < 10; ++i)
    {
      index.SearchWithOptGraph(attributes_query[i], query_load + i * dim, search_k, paras, res[i].data());
    }
  }

  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++)
  {
    index.SearchWithOptGraph(attributes_query[i], query_load + i * dim, search_k, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  int cnt = 0;
  for (unsigned i = 0; i < ground_num; i++)
  {
    for (unsigned j = 0; j < search_k; j++)
    {
      unsigned k = 0;
      for (; k < ground_dim; k++)
      {
        if (res[i][j] == ground_load[i * ground_dim + k])
          break;
      }
      if (k == ground_dim)
        cnt++;
    }
  }
  int act = 0;
  float acc = 1 - (float)cnt / (ground_num * search_k);
  std::cerr << "Search Time: " << diff.count() << " " << search_k << "NN accuracy: " << acc << " Distcount: " << act << std::endl;

  peak_memory_footprint();
  return 0;
}

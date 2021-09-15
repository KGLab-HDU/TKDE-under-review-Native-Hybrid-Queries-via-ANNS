#include <chrono>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim)
{ // load data with gist10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open())
  {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++)
  {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();
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

inline void load_result_data(char *filename, std::vector<std::vector<unsigned>> &res, unsigned &num, unsigned &dim)
{ 
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

void load_data_txt(char *filename, unsigned &num, unsigned &dim, std::vector<std::vector<int>> &data)
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

int main(int argc, char **argv)
{
  if (argc != 5)
  {
    std::cout << argv[0] << " graph_path data_path query_path groundtruth_path"
              << std::endl;
    exit(-1);
  }

  float *data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[2], data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);
  float *query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[3], query_load, query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  assert(dim == query_dim);

  char *groundtruth_file = argv[4];
  std::vector<std::vector<unsigned>> ground_load;
  unsigned ground_num, ground_dim;
  load_result_data(groundtruth_file, ground_load, ground_num, ground_dim);

  unsigned Search_K = 10;

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::FAST_L2,
                             (efanna2e::Index *)(&init_index));
  index.Load(argv[1]);
  index.OptimizeGraph(data_load);

  unsigned L = 100;
  
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  index.InitDistCount();
  std::vector<std::vector<unsigned>> res(query_num);

  for (unsigned i = 0; i < query_num; i++)
    res[i].resize(Search_K);

  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++)
  {
    index.SearchWithOptGraph(query_load + i * dim, Search_K, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> s_diff = e - s;
  size_t distcount = index.GetDistCount();

  int cnt = 0;
  for (unsigned i = 0; i < ground_num; i++)
  {
    for (unsigned j = 0; j < Search_K; j++)
    {
      unsigned k = 0;
      for (; k < Search_K; k++)
      {
        if (res[i][j] == ground_load[i][k])
          break;
      }
      if (k == Search_K)
        cnt++;
    }
  }
  float acc = 1 - (float)cnt / (ground_num * Search_K);
  std::cerr << "Search Time: " << s_diff.count() << " " << Search_K << "NN accuracy: " << acc << " Distcount: " << distcount << std::endl;

  peak_memory_footprint();
  return 0;
}

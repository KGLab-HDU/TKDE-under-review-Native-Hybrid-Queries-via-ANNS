#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <string>
#include <omp.h>

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

void load_data_txt(char *filename, unsigned &num, unsigned &dim, std::vector<std::vector<std::string>> &data)
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
  if (argc != 14)
  {
    std::cout << argv[0] << " data_file att_file save_graph save_attributetable K L iter S R Range PL B M"
              << std::endl;
    exit(-1);
  }
  omp_set_num_threads(64);

  char *data_path = argv[1];
  float *data_load = NULL;
  unsigned points_num, dim;
  efanna2e::load_data(data_path, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); //one must align the data before build

  char *label_data_path = argv[2];

  unsigned label_num, label_dim;
  std::vector<std::vector<string>> label_data;
  load_data_txt(label_data_path, label_num, label_dim, label_data);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *)(&init_index));
  for (int i = 0; i < points_num; i++)
  {
    index.AddAllNodeAttributes(label_data[i]);
  }
  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", atoi(argv[5]));
  paras.Set<unsigned>("L", atoi(argv[6]));
  paras.Set<unsigned>("iter", atoi(argv[7]));
  paras.Set<unsigned>("S", atoi(argv[8]));
  paras.Set<unsigned>("R", atoi(argv[9]));
  paras.Set<unsigned>("RANGE", atoi(argv[10]));
  paras.Set<unsigned>("PL", atoi(argv[11]));
  paras.Set<float>("B", atof(argv[12]));
  paras.Set<float>("M", atof(argv[13]));
  peak_memory_footprint();
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  peak_memory_footprint();
  std::chrono::duration<double> diff = e - s;
  std::cout << "Time cost: " << diff.count() << "\n";

  index.Save(argv[3]);
  index.SaveAttributeTable(argv[4]);

  return 0;
}

//
// Created by 付聪 on 2017/6/21.
//Modified by WMZ on 2020/6/6
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <omp.h>

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
  if (argc != 12)
  {
    std::cout << argv[0] << " data_file save_graph K L iter S R RANGE PL B M" << std::endl;
    exit(-1);
  }
  omp_set_num_threads(64);
  float *data_load = NULL;
  unsigned points_num, dim;
  efanna2e::load_data(argv[1], data_load, points_num, dim);
  char *graph_filename = argv[2];
  unsigned K = (unsigned)atoi(argv[3]);                         //开始构建的K近邻图, [50, 500], 间隔10
  unsigned L = (unsigned)atoi(argv[4]);                         //控制nn-descent的邻居集尺寸，不能小于K
  unsigned iter = (unsigned)atoi(argv[5]);                      //nn-descent迭代次数，[1, 20]，间隔 1
  unsigned S = (unsigned)atoi(argv[6]);                         //控制nn-descent 候选集尺寸
  unsigned R = (unsigned)atoi(argv[7]);                         //控制nn-descent反向邻居集的尺寸
  unsigned RANGE = (unsigned)atoi(argv[8]);                     //最终构建图的最大度数
  unsigned PL = (unsigned)atoi(argv[9]);                        //控制最后选边时候选集的大小
  float B = (float)atof(argv[10]);                              //控制加入选边候选集的邻居比例以及邻居的邻居比例，(0, 1]，间隔0.1
  float M = (float)atof(argv[11]);                              //选边参数，(0, 2]，间隔 0.1
  data_load = efanna2e::data_align(data_load, points_num, dim); //one must align the data before build

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *)(&init_index));

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("RANGE", RANGE);
  paras.Set<unsigned>("PL", PL);
  paras.Set<float>("B", B);
  paras.Set<float>("M", M);
  peak_memory_footprint();
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  peak_memory_footprint();
  std::chrono::duration<double> diff = e - s;
  std::cout << "Time cost: " << diff.count() << "\n";

  index.Save(graph_filename);

  return 0;
}

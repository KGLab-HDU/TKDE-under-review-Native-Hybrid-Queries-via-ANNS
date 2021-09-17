#ifndef EFANNA2E_INDEX_GRAPH_H
#define EFANNA2E_INDEX_GRAPH_H

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <boost/dynamic_bitset.hpp>

namespace efanna2e
{

  class IndexGraph : public Index
  {
  public:
    explicit IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer);

    virtual ~IndexGraph();

    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;
    void OptimizeGraph(float *data);
    void SearchWithOptGraph(std::vector<std::string> attributes,
                            const float *query, size_t K,
                            const Parameters &parameters,
                            unsigned *indices);
    void SearchWithOptGraph(std::vector<char> attribute,
                            const float *query, size_t K,
                            const Parameters &parameters,
                            unsigned *indices);
    size_t GetDistCount() { return dist_cout; }

    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

    virtual void Search(
        int query_id,
        const float *x,
        size_t k,
        const Parameters &parameters,
        unsigned *indices) override;
    void GraphAdd(const float *data, unsigned n, unsigned dim, const Parameters &parameters);
    void RefineGraph(const float *data, const Parameters &parameters);

    bool SaveAttributeTable(const std::string &fname) const;
    bool LoadAttributeTable(const std::string &fname);
    void AddAllNodeAttributes(std::vector<std::string> attributes);
    void statistic()
    {
      int sum = 0;
      int summ = 0;
      int sum1 = 0;
      int sum2 = 0;
      int sum3 = 0;
      int sum4 = 0;
      int sum5 = 0;
      int sum6 = 0;
      int sum7 = 0;

      int size1 = 0;
      int size2 = 0;
      int size3 = 0;
      int size4 = 0;
      int size5 = 0;
      int size6 = 0;
      int size7 = 0;
      int size8 = 0;

      for (int i = 0; i < nd_; i++)
      {
        int tsum = 0;
        int size = final_graph_[i].size();
        for (int j = 0; j < size; ++j)
        {
          int flag = 1;
          int id = final_graph_[i][j];
          summ++;
          for (int k = 0; k < attribute_number_; k++)
          {
            if (attributes_[i][k] != attributes_[id][k])
            {
              flag = 0;
              break;
            }
          }
          if (flag)
          {
            tsum++;
            sum++;
          }
        }
        float percent = (float)tsum / size;
        if (percent > 0 && percent < 0.2)
        {
          sum1++;
        }
        if (percent >= 0.2 && percent < 0.4)
        {
          sum2++;
        }
        if (percent >= 0.4 && percent < 0.6)
        {
          sum3++;
        }
        if (percent >= 0.6 && percent < 0.8)
        {
          sum4++;
        }
        if (percent >= 0.8 && percent < 1)
        {
          sum5++;
        }
        if (percent == 0)
        {
          sum6++;
        }
        if (percent == 1)
        {
          sum7++;
        }
        float percent2 = (float)size / width;
        if (percent2 > 0 && percent2 < 0.2)
        {
          size1++;
        }
        if (percent2 >= 0.2 && percent2 < 0.4)
        {
          size2++;
        }
        if (percent2 >= 0.4 && percent2 < 0.6)
        {
          size3++;
        }
        if (percent2 >= 0.6 && percent2 < 0.8)
        {
          size4++;
        }
        if (percent2 >= 0.8 && percent2 < 1)
        {
          size5++;
        }
        if (percent2 == 0)
        {
          size6++;
        }
        if (percent2 == 1)
        {
          size7++;
        }
      }
      std::cout << "0          " << sum6 << std::endl;
      std::cout << "0.0--0.2   " << sum1 << std::endl;
      std::cout << "0.2--0.4   " << sum2 << std::endl;
      std::cout << "0.4--0.6   " << sum3 << std::endl;
      std::cout << "0.6--0.8   " << sum4 << std::endl;
      std::cout << "0.8--1.0   " << sum5 << std::endl;
      std::cout << "1          " << sum7 << std::endl;
      std::cout << sum << "|" << summ << std::endl;
      std::cout << (float)sum / summ << std::endl;
      std::cout << "0          " << size6 << std::endl;
      std::cout << "0.0--0.2   " << size1 << std::endl;
      std::cout << "0.2--0.4   " << size2 << std::endl;
      std::cout << "0.4--0.6   " << size3 << std::endl;
      std::cout << "0.6--0.8   " << size4 << std::endl;
      std::cout << "0.8--1.0   " << size5 << std::endl;
      std::cout << "1          " << size7 << std::endl;
    };
    void reset_distcount()
    {
      this->dist_cout = 0;
    };

  protected:
    typedef std::vector<nhood> KNNGraph;
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    typedef std::vector<LockNeighbor> LockGraph;

    Index *initializer_;
    KNNGraph graph_;
    CompactGraph final_graph_;

    std::vector<std::vector<char>> attributes_;
    std::vector<std::vector<std::string>> attributes_code;
    int attribute_number_ = 3;

  private:
    void InitializeGraph(const Parameters &parameters);
    void InitializeGraph_Refine(const Parameters &parameters);
    void NNDescent(const Parameters &parameters);
    void join();
    void update(const Parameters &parameters);
    void Cut_Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);
    void get_neighbors(const unsigned q, const Parameters &parameter,
                       std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flags);
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset,
                       boost::dynamic_bitset<> cflags);
    void sync_prune(unsigned q, std::vector<Neighbor> &pool, float m,
                    const Parameters &parameters, SimpleNeighbor *cut_graph_);
    void InterInsert(unsigned n, unsigned range, float m,
                     std::vector<std::mutex> &locks,
                     SimpleNeighbor *cut_graph_);
    void DFS_expand(const Parameters &parameter);
    void get_cluster_center(const Parameters &parameter, boost::dynamic_bitset<> flags, unsigned &cc);
    void generate_control_set(std::vector<unsigned> &c,
                              std::vector<std::vector<unsigned>> &v,
                              unsigned N);
    float eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned>> &acc_eval_set);
    void get_neighbor_to_add(const float *point, const Parameters &parameters, LockGraph &g,
                             std::mt19937 &rng, std::vector<Neighbor> &retset, unsigned n_total);
    void compact_to_Lockgraph(LockGraph &g);
    void parallel_graph_insert(unsigned id, Neighbor nn, LockGraph &g, size_t K);
    //  void strong_connect(const Parameters &parameter);
    //  void DFS(boost::dynamic_bitset<> &flag,
    //           std::vector<std::pair<unsigned, unsigned>> &edges,
    //           unsigned root, unsigned &cnt);
    //  void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
    //                const Parameters &parameter);
    //  bool check_edge(unsigned h, unsigned t);
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset);
    void fusion_distance(float &dist, float &cnt);
    unsigned width;
    std::vector<unsigned> eps_;

    std::vector<char> Attribute2int(std::vector<std::string> str);
  };
}

#endif //EFANNA2E_INDEX_GRAPH_H

//
// Copyright (c) 2017 ZJULearning. All rights reserved.
// Modified by WMZ on 2020/6/16.
// This source code is licensed under the MIT license.
//

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


namespace efanna2e {

class IndexGraph : public Index {
 public:
  explicit IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexGraph();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;
  void OptimizeGraph(float *data);
  void SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters,
                                  unsigned *indices);
  size_t GetDistCount(){return dist_cout;}
  void InitDistCount() { dist_cout = 0; }
  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;
  void Graph_quality(std::vector<std::vector<unsigned>> &data);
  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void GraphAdd(const float* data, unsigned n, unsigned dim, const Parameters &parameters);
  void RefineGraph(const float* data, const Parameters &parameters);

 protected:
  typedef std::vector<nhood> KNNGraph;
  typedef std::vector<std::vector<unsigned > > CompactGraph;
  typedef std::vector<LockNeighbor > LockGraph;

  Index *initializer_;
  KNNGraph graph_;
  CompactGraph final_graph_;



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
                                      std::vector<std::vector<unsigned> > &v,
                                      unsigned N);
  float eval_recall(std::vector<unsigned>& ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);
  void get_neighbor_to_add(const float* point, const Parameters &parameters, LockGraph& g,
                           std::mt19937& rng, std::vector<Neighbor>& retset, unsigned n_total);
  void compact_to_Lockgraph(LockGraph &g);
  void parallel_graph_insert(unsigned id, Neighbor nn, LockGraph& g, size_t K);

  unsigned width;
  std::vector<unsigned> eps_;

};

}

#endif //EFANNA2E_INDEX_GRAPH_H

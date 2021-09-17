#include <efanna2e/index_graph.h>
#include <efanna2e/exceptions.h>
#include <efanna2e/parameters.h>
#include <omp.h>
#include <set>
#include <queue>
#include <stack>

namespace efanna2e
{
#define _CONTROL_NUM 100
  IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer)
      : Index(dimension, n, m),
        initializer_{initializer}
  {
    assert(dimension == initializer->GetDimension());
  }
  IndexGraph::~IndexGraph() { std::cout << "release index." << std::endl; }

  void IndexGraph::join()
  {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; n++)
    {
      graph_[n].join([&](unsigned i, unsigned j)
                     {
                       if (i != j)
                       {
                         float dist = distance_->compare(data_ + i * dimension_, data_ + j * dimension_, dimension_);

                         float cnt = 0;
                         for (int k = 0; k < attribute_number_; k++)
                         {
                           if (attributes_[i][k] != attributes_[j][k])
                           {
                             cnt++;
                           }
                         }
                         fusion_distance(dist, cnt);

                         graph_[i].insert(j, dist);
                         graph_[j].insert(i, dist);
                       }
                     });
    }
  }
  void IndexGraph::update(const Parameters &parameters)
  {
    unsigned S = parameters.Get<unsigned>("S");
    unsigned R = parameters.Get<unsigned>("R");
    unsigned L = parameters.Get<unsigned>("L");
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; i++)
    {
      std::vector<unsigned>().swap(graph_[i].nn_new);
      std::vector<unsigned>().swap(graph_[i].nn_old);
      //std::vector<unsigned>().swap(graph_[i].rnn_new);
      //std::vector<unsigned>().swap(graph_[i].rnn_old);
      //graph_[i].nn_new.clear();
      //graph_[i].nn_old.clear();
      //graph_[i].rnn_new.clear();
      //graph_[i].rnn_old.clear();
    }
#pragma omp parallel for
    for (unsigned n = 0; n < nd_; ++n)
    {
      auto &nn = graph_[n];
      // std::sort(nn.pool.begin(), nn.pool.end());
      if (nn.pool.size() > L)
        nn.pool.resize(L);
      nn.pool.reserve(L + 1);
      unsigned maxl = std::min(nn.M + S, (unsigned)nn.pool.size());
      unsigned c = 0;
      unsigned l = 0;
      //std::sort(nn.pool.begin(), nn.pool.end());
      //if(n==0)std::cout << nn.pool[0].distance<<","<< nn.pool[1].distance<<","<< nn.pool[2].distance<< std::endl;
      while ((l < maxl) && (c < S))
      {
        if (nn.pool[l].flag)
          ++c;
        ++l;
      }
      nn.M = l;
    }
#pragma omp parallel for
    for (unsigned n = 0; n < nd_; ++n)
    {
      auto &nnhd = graph_[n];
      auto &nn_new = nnhd.nn_new;
      auto &nn_old = nnhd.nn_old;
      for (unsigned l = 0; l < nnhd.M; ++l)
      {
        auto &nn = nnhd.pool[l];
        auto &nhood_o = graph_[nn.id]; // nn on the other side of the edge
        if (nn.flag)
        {
          nn_new.push_back(nn.id);
          if (nn.distance > nhood_o.pool.back().distance)
          {
            LockGuard guard(nhood_o.lock);
            if (nhood_o.rnn_new.size() < R)
              nhood_o.rnn_new.push_back(n);
            else
            {
              unsigned int pos = rand() % R;
              nhood_o.rnn_new[pos] = n;
            }
          }
          nn.flag = false;
        }
        else
        {
          nn_old.push_back(nn.id);
          if (nn.distance > nhood_o.pool.back().distance)
          {
            LockGuard guard(nhood_o.lock);
            if (nhood_o.rnn_old.size() < R)
              nhood_o.rnn_old.push_back(n);
            else
            {
              unsigned int pos = rand() % R;
              nhood_o.rnn_old[pos] = n;
            }
          }
        }
      }
      // std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
    }
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; ++i)
    {
      auto &nn_new = graph_[i].nn_new;
      auto &nn_old = graph_[i].nn_old;
      auto &rnn_new = graph_[i].rnn_new;
      auto &rnn_old = graph_[i].rnn_old;
      if (R && rnn_new.size() > R)
      {
        std::random_shuffle(rnn_new.begin(), rnn_new.end());
        rnn_new.resize(R);
      }
      nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
      if (R && rnn_old.size() > R)
      {
        std::random_shuffle(rnn_old.begin(), rnn_old.end());
        rnn_old.resize(R);
      }
      nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
      if (nn_old.size() > R * 2)
      {
        nn_old.resize(R * 2);
        nn_old.reserve(R * 2);
      }
      std::vector<unsigned>().swap(graph_[i].rnn_new);
      std::vector<unsigned>().swap(graph_[i].rnn_old);
    }
  } //update

  void IndexGraph::NNDescent(const Parameters &parameters)
  {
    unsigned iter = parameters.Get<unsigned>("iter");
    std::mt19937 rng(rand());
    std::vector<unsigned> control_points(_CONTROL_NUM);
    std::vector<std::vector<unsigned>> acc_eval_set(_CONTROL_NUM);
    GenRandom(rng, &control_points[0], control_points.size(), nd_);
    generate_control_set(control_points, acc_eval_set, nd_);
    for (unsigned it = 0; it < iter; it++)
    {
      join();
      update(parameters);
      //checkDup();
      float acc = eval_recall(control_points, acc_eval_set);
      std::cout << "iter: " << it << std::endl;
      if (acc >= 0.8)
        break;
    }
  }

  void IndexGraph::Cut_Link(const Parameters &parameters, SimpleNeighbor *cut_graph_)
  {
    unsigned range = parameters.Get<unsigned>("RANGE");
    float m = parameters.Get<float>("M");
    std::vector<std::mutex> locks(nd_);

    // std::cout << "test1\n";
#pragma omp parallel
    {
      std::vector<Neighbor> pool;
#pragma omp for schedule(dynamic, 100)
      for (unsigned n = 0; n < nd_; ++n)
      {
        pool.clear();
        sync_prune(n, pool, m, parameters, cut_graph_); //cut edge
      }
#pragma omp for schedule(dynamic, 100)
      for (unsigned n = 0; n < nd_; ++n)
      {
        InterInsert(n, range, m, locks, cut_graph_); //reverse connection
      }
    }
  }

  void IndexGraph::get_neighbors(const float *query, const Parameters &parameter,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset,
                                 boost::dynamic_bitset<> cflags)
  {
    unsigned L = parameter.Get<unsigned>("PL");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

    boost::dynamic_bitset<> flags{nd_, 0};
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_ || !cflags[id])
        continue;
      // std::cout<<id<<std::endl;
      float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                      (unsigned)dimension_);

      float cnt = 0;
      for (int k = 0; k < attribute_number_; k++)
      {
        cnt += (float)(attributes_code[k].size() - 1) / (float)(attributes_code[k].size());
      }
      fusion_distance(dist, cnt);

      retset[i] = Neighbor(id, dist, true);
      flags[id] = 1;
      L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m)
        {
          unsigned id = final_graph_[n][m];
          if (flags[id])
            continue;
          flags[id] = 1;

          float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                          (unsigned)dimension_);

          float cnt = 0;
          for (int k = 0; k < attribute_number_; k++)
          {
            cnt += (float)(attributes_code[k].size() - 1) / (float)(attributes_code[k].size());
          }
          fusion_distance(dist, cnt);

          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[L - 1].distance)
            continue;
          int r = InsertIntoPool(retset.data(), L, nn);

          if (L + 1 < retset.size())
            ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
  }

  void IndexGraph::get_neighbors(const unsigned q, const Parameters &parameter,
                                 std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flagss)
  {
    boost::dynamic_bitset<> flags{nd_, 0};
    unsigned PL = parameter.Get<unsigned>("PL");
    unsigned K = parameter.Get<unsigned>("K");
    unsigned b = parameter.Get<float>("B");
    unsigned ML = PL + pool.size();
    float bK = (float)K * b;
    flags[q] = true;
    for (unsigned i = 0; i < graph_[q].pool.size() && i < bK; i++)
    {
      unsigned nid = graph_[q].pool[i].id;
      for (unsigned nn = 0; nn < graph_[nid].pool.size() && i < bK; nn++)
      {
        unsigned nnid = graph_[nid].pool[nn].id;
        if (flags[nnid] || flagss[nnid])
          continue;
        flags[nnid] = true;
        float d1 = graph_[q].pool[i].distance;
        float d2 = graph_[nid].pool[nn].distance;
        if (d1 < d2 && d2 - d1 > d1)
          continue;
        float dist = distance_->compare(data_ + dimension_ * q,
                                        data_ + dimension_ * nnid, dimension_);

        float cnt = 0;
        for (int k = 0; k < attribute_number_; k++)
        {
          if (attributes_[q][k] != attributes_[nnid][k])
          {
            cnt++;
          }
        }
        fusion_distance(dist, cnt);

        pool.push_back(Neighbor(nnid, dist, true));
        if (pool.size() >= ML)
          break;
      }
      if (pool.size() >= ML)
        break;
    }
    // if (pool.size() > ML) pool.resize(ML);
  }

  void IndexGraph::get_neighbors(const float *query, const Parameters &parameter,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset)
  {
    unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

    boost::dynamic_bitset<> flags{nd_, 0};
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      // std::cout<<id<<std::endl;
      float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                      (unsigned)dimension_);
      retset[i] = Neighbor(id, dist, true);
      flags[id] = 1;
      L++;
    }
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m)
        {
          unsigned id = final_graph_[n][m];
          if (flags[id])
            continue;
          flags[id] = 1;

          float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                          (unsigned)dimension_);
          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[L - 1].distance)
            continue;
          int r = InsertIntoPool(retset.data(), L, nn);

          if (L + 1 < retset.size())
            ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
  }

  void IndexGraph::sync_prune(unsigned q, std::vector<Neighbor> &pool, float m,
                              const Parameters &parameters, SimpleNeighbor *cut_graph_)
  {
    unsigned range = parameters.Get<unsigned>("RANGE");
    width = range;
    unsigned start = 0;

    boost::dynamic_bitset<> flags{nd_, 0};
    for (unsigned nn = 0; nn < graph_[q].pool.size(); nn++)
    {
      unsigned id = graph_[q].pool[nn].id;
      flags[id] = 1;
      float dist = graph_[q].pool[nn].distance;
      bool f = graph_[q].pool[nn].flag;
      pool.push_back(Neighbor(id, dist, f));
    }
    get_neighbors(q, parameters, pool, flags);
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    if (pool[start].id == q)
      start++;
    result.push_back(pool[start]);

    while (result.size() < range && (++start) < pool.size())
    {
      auto &p = pool[start];
      bool occlude = false;
      for (unsigned t = 0; t < result.size(); t++)
      {
        if (p.id == result[t].id)
        {
          occlude = true;
          break;
        }
        float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                       data_ + dimension_ * (size_t)p.id,
                                       (unsigned)dimension_);

        float cnt = 0;
        for (int k = 0; k < attribute_number_; k++)
        {
          if (attributes_[result[t].id][k] != attributes_[p.id][k])
          {
            cnt++;
          }
        }
        fusion_distance(djk, cnt);

        // float cos_ij = (p.distance + result[t].distance - djk) / 2 /
        //                sqrt(p.distance * result[t].distance);
        // if (cos_ij > threshold) {
        //   occlude = true;
        //   break;
        // }
        if (m * djk < p.distance)
        {
          occlude = true;
          break;
        }
      }
      if (!occlude)
        result.push_back(p);
    }
    SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
    for (size_t t = 0; t < result.size(); t++)
    {
      des_pool[t].id = result[t].id;
      des_pool[t].distance = result[t].distance;
    }
    if (result.size() < range)
    {
      des_pool[result.size()].distance = -1;
    }
  }

  void IndexGraph::InterInsert(unsigned n, unsigned range, float m,
                               std::vector<std::mutex> &locks,
                               SimpleNeighbor *cut_graph_)
  {
    SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
    for (size_t i = 0; i < range; i++)
    {
      if (src_pool[i].distance == -1)
        break;

      SimpleNeighbor sn(n, src_pool[i].distance);
      size_t des = src_pool[i].id;
      SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

      std::vector<SimpleNeighbor> temp_pool;
      int dup = 0;
      {
        LockGuard guard(locks[des]);
        for (size_t j = 0; j < range; j++)
        {
          if (des_pool[j].distance == -1)
            break;
          if (n == des_pool[j].id)
          {
            dup = 1;
            break;
          }
          temp_pool.push_back(des_pool[j]);
        }
      }
      if (dup)
        continue;

      temp_pool.push_back(sn);
      if (temp_pool.size() > range)
      {
        std::vector<SimpleNeighbor> result;
        unsigned start = 0;
        std::sort(temp_pool.begin(), temp_pool.end());
        result.push_back(temp_pool[start]);
        while (result.size() < range && (++start) < temp_pool.size())
        {
          auto &p = temp_pool[start];
          bool occlude = false;
          for (unsigned t = 0; t < result.size(); t++)
          {
            if (p.id == result[t].id)
            {
              occlude = true;
              break;
            }
            float djk = distance_->compare(
                data_ + dimension_ * (size_t)result[t].id,
                data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);

            float cnt = 0;
            for (int k = 0; k < attribute_number_; k++)
            {
              if (attributes_[result[t].id][k] != attributes_[p.id][k])
              {
                cnt++;
              }
            }
            fusion_distance(djk, cnt);

            if (djk < p.distance)
            {
              occlude = true;
              break;
            }
          }
          if (!occlude)
            result.push_back(p);
        }
        {
          LockGuard guard(locks[des]);
          for (unsigned t = 0; t < result.size(); t++)
          {
            des_pool[t] = result[t];
          }
          if (result.size() < range)
          {
            des_pool[result.size()].distance = -1;
          }
        }
      }
      else
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < range; t++)
        {
          if (des_pool[t].distance == -1)
          {
            des_pool[t] = sn;
            if (t + 1 < range)
              des_pool[t + 1].distance = -1;
            break;
          }
        }
      }
    }
  }

  void IndexGraph::get_cluster_center(const Parameters &parameter, boost::dynamic_bitset<> cflags, unsigned &cc)
  {
    float *center = new float[dimension_];
    for (unsigned j = 0; j < dimension_; j++)
      center[j] = 0;
    for (unsigned i = 0; i < nd_; i++)
    {
      if (cflags[i])
      {
        for (unsigned j = 0; j < dimension_; j++)
        {
          center[j] += data_[i * dimension_ + j];
        }
      }
    }
    for (unsigned j = 0; j < dimension_; j++)
    {
      center[j] /= nd_;
    }
    std::vector<Neighbor> tmp, pool;
    cc = rand() % nd_; // random initialize navigating point
    get_neighbors(center, parameter, tmp, pool, cflags);
    cc = tmp[0].id;
  }

  void IndexGraph::DFS_expand(const Parameters &parameter)
  { //partition dataset
    // unsigned n_try = parameter.Get<unsigned>("n_try");
    // unsigned range = parameter.Get<unsigned>("RANGE");
    unsigned id = rand() % nd_;

    boost::dynamic_bitset<> flags{nd_, 0};
    boost::dynamic_bitset<> cflags{nd_, 0};
    std::queue<unsigned> myqueue;
    myqueue.push(id);
    flags[id] = true;
    cflags[id] = true;

    std::vector<unsigned> uncheck_set(1);

    while (uncheck_set.size() > 0)
    {

      while (!myqueue.empty())
      {
        unsigned q_front = myqueue.front();
        myqueue.pop();

        for (unsigned j = 0; j < final_graph_[q_front].size(); j++)
        {
          unsigned child = final_graph_[q_front][j];
          if (flags[child])
            continue;
          flags[child] = true;
          cflags[child] = true;
          myqueue.push(child);
        }
      }
      unsigned cc;
      get_cluster_center(parameter, cflags, cc);
      eps_.push_back(cc);

      uncheck_set.clear();
      for (unsigned j = 0; j < nd_; j++)
      {
        if (flags[j])
          continue;
        uncheck_set.push_back(j);
      }
      //std::cout <<i<<":"<< uncheck_set.size() << '\n';
      if (uncheck_set.size() > 0)
      {
        // for(unsigned j=0; j<nd_; j++){
        //   if(flags[j] && final_graph_[j].size()<range){
        //     final_graph_[j].push_back(uncheck_set[0]);
        //     break;
        //   }
        // }
        myqueue.push(uncheck_set[0]);
        flags[uncheck_set[0]] = true;
      }
      cflags.resize(nd_, 0);
    }
    std::cout << "navigation_points: " << eps_.size() << "\n";
  }

  void IndexGraph::generate_control_set(std::vector<unsigned> &c,
                                        std::vector<std::vector<unsigned>> &v,
                                        unsigned N)
  {
#pragma omp parallel for
    for (unsigned i = 0; i < c.size(); i++)
    {
      std::vector<Neighbor> tmp;
      for (unsigned j = 0; j < N; j++)
      {
        float dist = distance_->compare(data_ + c[i] * dimension_, data_ + j * dimension_, dimension_);

        float cnt = 0;
        for (int k = 0; k < attribute_number_; k++)
        {
          if (attributes_[c[i]][k] != attributes_[j][k])
          {
            cnt++;
          }
        }
        fusion_distance(djk, cnt);

        tmp.push_back(Neighbor(j, dist, true));
      }
      std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
      for (unsigned j = 0; j < _CONTROL_NUM; j++)
      {
        v[i].push_back(tmp[j].id);
      }
    }
  }

  float IndexGraph::eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned>> &acc_eval_set)
  {
    float mean_acc = 0;
    for (unsigned i = 0; i < ctrl_points.size(); i++)
    {
      float acc = 0;
      auto &g = graph_[ctrl_points[i]].pool;
      auto &v = acc_eval_set[i];
      for (unsigned j = 0; j < g.size(); j++)
      {
        for (unsigned k = 0; k < v.size(); k++)
        {
          if (g[j].id == v[k])
          {
            acc++;
            break;
          }
        }
      }
      mean_acc += acc / v.size();
    }
    float ret_recall = mean_acc / ctrl_points.size();
    std::cout << "recall : " << ret_recall << std::endl;
    return ret_recall;
  }

  void IndexGraph::InitializeGraph(const Parameters &parameters)
  {

    const unsigned L = parameters.Get<unsigned>("L");
    const unsigned S = parameters.Get<unsigned>("S");

    graph_.reserve(nd_);
    std::mt19937 rng(rand());
    for (unsigned i = 0; i < nd_; i++)
    {
      graph_.push_back(nhood(L, S, rng, (unsigned)nd_));
    }
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; i++)
    {
      //const float *query = data_ + i * dimension_;
      std::vector<unsigned> tmp(S + 1);
      initializer_->Search(i, data_, S + 1, parameters, tmp.data());

      for (unsigned j = 0; j < S; j++)
      {
        unsigned id = tmp[j];
        if (id == i)
          continue;
        float dist = distance_->compare(data_ + i * dimension_, data_ + id * dimension_, (unsigned)dimension_);

        float cnt = 0;
        for (int k = 0; k < attribute_number_; k++)
        {
          if (attributes_[i][k] != attributes_[id][k])
          {
            cnt++;
          }
        }
        fusion_distance(dist, cnt);

        graph_[i].pool.push_back(Neighbor(id, dist, true));
      }
      std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
      graph_[i].pool.reserve(L + 1);
    }
  }

  void IndexGraph::InitializeGraph_Refine(const Parameters &parameters)
  {
    assert(final_graph_.size() == nd_);

    const unsigned L = parameters.Get<unsigned>("L");
    const unsigned S = parameters.Get<unsigned>("S");

    graph_.reserve(nd_);
    std::mt19937 rng(rand());
    for (unsigned i = 0; i < nd_; i++)
    {
      graph_.push_back(nhood(L, S, rng, (unsigned)nd_));
    }
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; i++)
    {
      auto &ids = final_graph_[i];
      std::sort(ids.begin(), ids.end());

      size_t K = ids.size();

      for (unsigned j = 0; j < K; j++)
      {
        unsigned id = ids[j];
        if (id == i || (j > 0 && id == ids[j - 1]))
          continue;
        float dist = distance_->compare(data_ + i * dimension_, data_ + id * dimension_, (unsigned)dimension_);

        float cnt = 0;
        for (int k = 0; k < attribute_number_; k++)
        {
          if (attributes_[i][k] != attributes_[id][k])
          {
            cnt++;
          }
        }
        fusion_distance(dist, cnt);

        graph_[i].pool.push_back(Neighbor(id, dist, true));
      }
      std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
      graph_[i].pool.reserve(L);
      std::vector<unsigned>().swap(ids);
    }
    CompactGraph().swap(final_graph_);
  }

  void IndexGraph::RefineGraph(const float *data, const Parameters &parameters)
  {
    data_ = data;
    assert(initializer_->HasBuilt());

    InitializeGraph_Refine(parameters);
    NNDescent(parameters);

    final_graph_.reserve(nd_);
    std::cout << nd_ << std::endl;
    unsigned K = parameters.Get<unsigned>("K");
    for (unsigned i = 0; i < nd_; i++)
    {
      std::vector<unsigned> tmp;
      std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
      for (unsigned j = 0; j < K; j++)
      {
        tmp.push_back(graph_[i].pool[j].id);
      }
      tmp.reserve(K);
      final_graph_.push_back(tmp);
      std::vector<Neighbor>().swap(graph_[i].pool);
      std::vector<unsigned>().swap(graph_[i].nn_new);
      std::vector<unsigned>().swap(graph_[i].nn_old);
      std::vector<unsigned>().swap(graph_[i].rnn_new);
      std::vector<unsigned>().swap(graph_[i].rnn_new);
    }
    std::vector<nhood>().swap(graph_);
    has_built = true;
  }

  void IndexGraph::Build(size_t n, const float *data, const Parameters &parameters)
  {

    //assert(initializer_->GetDataset() == data);
    data_ = data;
    assert(initializer_->HasBuilt());
    unsigned range = parameters.Get<unsigned>("RANGE");
    InitializeGraph(parameters);
    NNDescent(parameters);
    SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
    Cut_Link(parameters, cut_graph_);
    final_graph_.resize(nd_);

    for (size_t i = 0; i < nd_; i++)
    {
      SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
      unsigned pool_size = 0;
      for (unsigned j = 0; j < range; j++)
      {
        if (pool[j].distance == -1)
          break;
        pool_size = j;
      }
      pool_size++;
      final_graph_[i].resize(pool_size);
      for (unsigned j = 0; j < pool_size; j++)
      {
        final_graph_[i][j] = pool[j].id;
      }
      std::vector<Neighbor>().swap(graph_[i].pool);
      std::vector<unsigned>().swap(graph_[i].nn_new);
      std::vector<unsigned>().swap(graph_[i].nn_old);
      std::vector<unsigned>().swap(graph_[i].rnn_new);
      std::vector<unsigned>().swap(graph_[i].rnn_new);
    }
    std::vector<nhood>().swap(graph_);
    //RefineGraph(parameters);

    //DFS_expand(parameters);
    unsigned max, min, avg;
    max = 0;
    min = nd_;
    avg = 0;
    for (size_t i = 0; i < nd_; i++)
    {
      auto size = final_graph_[i].size();
      max = max < size ? size : max;
      min = min > size ? size : min;
      avg += size;
    }
    avg /= 1.0 * nd_;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n",
           max, min, avg);

    //std::cout << "connect..." << std::endl;
    //strong_connect(parameters);
    //max = 0;
    //min = nd_;
    //avg = 0;
    //for (size_t i = 0; i < nd_; i++)
    //{
    //  auto size = final_graph_[i].size();
    //  max = max < size ? size : max;
    //  min = min > size ? size : min;
    //  avg += size;
    //}
    //avg /= 1.0 * nd_;
    //printf("Degree Statistics(After TreeGrow): Max = %d, Min = %d, Avg = %d\n",
    //       max, min, avg);
    //DFS_expand(parameters);
    has_built = true;
  }

  //  void IndexGraph::strong_connect(const Parameters &parameter)
  //  {
  //    unsigned n_try = parameter.Get<unsigned>("n_try");
  //    std::vector<std::pair<unsigned, unsigned>> edges_all;
  //    std::mutex edge_lock;
  //#pragma omp parallel for
  //    for (unsigned nt = 0; nt < n_try; nt++)
  //    {
  //      unsigned root = rand() % nd_;
  //      boost::dynamic_bitset<> flags{nd_, 0};
  //      unsigned unlinked_cnt = 0;
  //      std::vector<std::pair<unsigned, unsigned>> edges;
  //      while (unlinked_cnt < nd_)
  //      {
  //        DFS(flags, edges, root, unlinked_cnt);
  //        //std::cout << unlinked_cnt << '\n';
  //        if (unlinked_cnt >= nd_)
  //          break;
  //        findroot(flags, root, parameter);
  //        //std::cout << "new root"<<":"<<root << '\n';
  //      }
  //
  //      LockGuard guard(edge_lock);
  //
  //      for (unsigned i = 0; i < edges.size(); i++)
  //      {
  //        edges_all.push_back(edges[i]);
  //      }
  //    }
  //    unsigned ecnt = 0;
  //    for (unsigned e = 0; e < edges_all.size(); e++)
  //    {
  //      unsigned start = edges_all[e].first;
  //      unsigned end = edges_all[e].second;
  //      unsigned flag = 1;
  //      for (unsigned j = 0; j < final_graph_[start].size(); j++)
  //      {
  //        if (end == final_graph_[start][j])
  //        {
  //          flag = 0;
  //        }
  //      }
  //      if (flag)
  //      {
  //        final_graph_[start].push_back(end);
  //        ecnt++;
  //      }
  //    }
  //    for (size_t i = 0; i < nd_; ++i)
  //    {
  //      if (final_graph_[i].size() > width)
  //      {
  //        width = final_graph_[i].size();
  //      }
  //    }
  //  }
  //
  //  void IndexGraph::DFS(boost::dynamic_bitset<> &flag,
  //                       std::vector<std::pair<unsigned, unsigned>> &edges,
  //                       unsigned root, unsigned &cnt)
  //  {
  //    unsigned tmp = root;
  //    std::stack<unsigned> s;
  //    s.push(root);
  //    if (!flag[root])
  //      cnt++;
  //    flag[root] = true;
  //    while (!s.empty())
  //    {
  //      unsigned next = nd_ + 1;
  //      for (unsigned i = 0; i < final_graph_[tmp].size(); i++)
  //      {
  //        if (flag[final_graph_[tmp][i]] == false)
  //        {
  //          next = final_graph_[tmp][i];
  //          break;
  //        }
  //      }
  //      // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
  //      if (next == (nd_ + 1))
  //      {
  //        unsigned head = s.top();
  //        s.pop();
  //        if (s.empty())
  //          break;
  //        tmp = s.top();
  //        unsigned tail = tmp;
  //        if (check_edge(head, tail))
  //        {
  //          edges.push_back(std::make_pair(head, tail));
  //        }
  //        continue;
  //      }
  //      tmp = next;
  //      flag[tmp] = true;
  //      s.push(tmp);
  //      cnt++;
  //    }
  //  }
  //
  //  void IndexGraph::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
  //                            const Parameters &parameter)
  //  {
  //    unsigned id = nd_;
  //    for (unsigned i = 0; i < nd_; i++)
  //    {
  //      if (flag[i] == false)
  //      {
  //        id = i;
  //        break;
  //      }
  //    }
  //
  //    if (id == nd_)
  //      return; // No Unlinked Node
  //
  //    std::vector<Neighbor> tmp, pool;
  //    get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  //    std::sort(pool.begin(), pool.end());
  //
  //    bool found = false;
  //    for (unsigned i = 0; i < pool.size(); i++)
  //    {
  //      if (flag[pool[i].id])
  //      {
  //        // std::cout << pool[i].id << '\n';
  //        root = pool[i].id;
  //        found = true;
  //        break;
  //      }
  //    }
  //    if (!found)
  //    {
  //      for (int retry = 0; retry < 1000; ++retry)
  //      {
  //        unsigned rid = rand() % nd_;
  //        if (flag[rid])
  //        {
  //          root = rid;
  //          break;
  //        }
  //      }
  //    }
  //    final_graph_[root].push_back(id);
  //  }
  //
  //  bool IndexGraph::check_edge(unsigned h, unsigned t)
  //  {
  //    bool flag = true;
  //    for (unsigned i = 0; i < final_graph_[h].size(); i++)
  //    {
  //      if (t == final_graph_[h][i])
  //        flag = false;
  //    }
  //    return flag;
  //  }

  void IndexGraph::Search(
      int query_id,
      const float *x,
      size_t K,
      const Parameters &parameter,
      unsigned *indices)
  {
    const unsigned L = parameter.Get<unsigned>("L_search");

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

    std::vector<char> flags(nd_);
    memset(flags.data(), 0, nd_ * sizeof(char));
    for (unsigned i = 0; i < L; i++)
    {
      unsigned id = init_ids[i];
      float dist = distance_->compare(data_ + dimension_ * id, data_ + dimension_ * query_id, (unsigned)dimension_);

      float cnt = 0;
      for (int k = 0; k < attribute_number_; k++)
      {
        if (attributes_[query_id][k] != attributes_[id][k])
        {
          cnt++;
        }
      }
      fusion_distance(dist, cnt);

      retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m)
        {
          unsigned id = final_graph_[n][m];
          if (flags[id])
            continue;
          flags[id] = 1;
          float dist = distance_->compare(data_ + dimension_ * query_id, data_ + dimension_ * id, (unsigned)dimension_);

          float cnt = 0;
          for (int k = 0; k < attribute_number_; k++)
          {
            if (attributes_[query_id][k] != attributes_[id][k])
            {
              cnt++;
            }
          }
          fusion_distance(dist, cnt);

          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          //if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
        //lock to here
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++)
    {
      indices[i] = retset[i].id;
    }
  }

  void IndexGraph::Save(const char *filename)
  {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph_.size() == nd_);

    out.write((char *)&width, sizeof(unsigned));
    unsigned n_ep = eps_.size();
    out.write((char *)&n_ep, sizeof(unsigned));
    out.write((char *)eps_.data(), n_ep * sizeof(unsigned));
    for (unsigned i = 0; i < nd_; i++)
    {
      unsigned GK = (unsigned)final_graph_[i].size();
      out.write((char *)&GK, sizeof(unsigned));
      out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
    }

    out.write((char *)&attribute_number_, sizeof(int));
    for (unsigned i = 0; i < nd_; i++)
    {
      out.write((char *)attributes_[i].data(), attribute_number_ * sizeof(char));
    }
    out.close();
  }

  void IndexGraph::Load(const char *filename)
  {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *)&width, sizeof(unsigned));
    unsigned n_ep = 0;
    in.read((char *)&n_ep, sizeof(unsigned));
    eps_.resize(n_ep);
    in.read((char *)eps_.data(), n_ep * sizeof(unsigned));
    // width=100;
    unsigned cc = 0;

    for (unsigned i = 0; i < nd_; i++)
    {
      unsigned k;
      in.read((char *)&k, sizeof(unsigned));
      cc += k;
      std::vector<unsigned> tmp(k);
      in.read((char *)tmp.data(), k * sizeof(unsigned));
      final_graph_.push_back(tmp);
    }

    in.read((char *)&attribute_number_, sizeof(int));
    while (!in.eof())
    {
      std::vector<char> tmp(attribute_number_);
      in.read((char *)tmp.data(), attribute_number_ * sizeof(char));
      if (in.eof())
        break;
      attributes_.push_back(tmp);
    }
    std::cout << "attribute dim:" << attribute_number_ << std::endl;
    std::cout << "attribute number:" << attributes_.size() << std::endl;
    cc /= nd_;
    std::cerr << "Average Degree = " << cc << std::endl;
    // statistic();
  }

  void IndexGraph::SearchWithOptGraph(std::vector<char> attribute,
                                      const float *query, size_t K,
                                      const Parameters &parameters,
                                      unsigned *indices)
  {
    unsigned L = parameters.Get<unsigned>("L_search");
    float weight_search = parameters.Get<float>("weight_search");
    DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
    // assert(eps_.size() < L);

    //for (unsigned i = 0; i < eps_.size() && i < L; i++)
    //{
    //  init_ids[i] = eps_[i];
    //}

    boost::dynamic_bitset<> flags{nd_, 0};
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      float *x = (float *)(opt_graph_ + node_size * id);
      float norm_x = *x;
      x++;
      float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);

      float cnt = 0;
      char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
      for (int k = 0; k < attribute_number_; k++)
      {
        if (id_attribute[k] != attribute[k])
        {
          cnt++;
        }
      }
      //dist += dist * cnt / (float)attribute_number_;
      dist += cnt * weight_search;

      // float d = distance_ -> compare(x, query, (unsigned)dimension_);
      // std::cout << d << std::endl;
      dist_cout++;
      retset[i] = Neighbor(id, dist, true);
      flags[id] = true;
      L++;
    }
    // std::cout<<L<<std::endl;

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len + attribute_len, _MM_HINT_T0);
        unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len + attribute_len);
        neighbors++;
        unsigned MaxM = *neighbors;
        neighbors++;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m)
        {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          float *data = (float *)(opt_graph_ + node_size * id);
          float norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned)dimension_);

          float cnt = 0;
          char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
          for (int k = 0; k < attribute_number_; k++)
          {
            if (id_attribute[k] != attribute[k])
            {
              cnt++;
            }
          }
          //dist += dist * cnt / (float)attribute_number_;
          dist += cnt * weight_search;

          dist_cout++;
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    // for (size_t i = 0; i < L; i++) {  
    //   retset[i].flag = true;
    // }
    k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (!retset[k].flag)
      {
        retset[k].flag = true;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len + attribute_len, _MM_HINT_T0);
        unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len + attribute_len);
        unsigned MaxM = *neighbors;
        neighbors += 2;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m)
        {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          float *data = (float *)(opt_graph_ + node_size * id);
          float norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned)dimension_);

          float cnt = 0;
          char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
          for (int k = 0; k < attribute_number_; k++)
          {
            if (id_attribute[k] != attribute[k])
            {
              cnt++;
            }
          }
          //dist += dist * cnt / (float)attribute_number_;
          dist += cnt * weight_search;

          dist_cout++;
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, false);
          int r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++)
    {
      indices[i] = retset[i].id;
    }
  }

  void IndexGraph::SearchWithOptGraph(std::vector<std::string> attributes,
                                      const float *query, size_t K,
                                      const Parameters &parameters,
                                      unsigned *indices)
  {
    std::vector<char> attribute = Attribute2int(attributes);
    if (attribute.size() != attribute_number_)
    {
      std::cout << "wrong attributes";
      return;
    }
    unsigned L = parameters.Get<unsigned>("L_search");
    float weight_search = parameters.Get<float>("weight_search");
    DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
    // assert(eps_.size() < L);

    //for (unsigned i = 0; i < eps_.size() && i < L; i++)
    //{
    //  init_ids[i] = eps_[i];
    //}

    boost::dynamic_bitset<> flags{nd_, 0};
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      float *x = (float *)(opt_graph_ + node_size * id);
      float norm_x = *x;
      x++;
      float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);

      float cnt = 0;
      char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
      for (int k = 0; k < attribute_number_; k++)
      {
        if (id_attribute[k] != attribute[k])
        {
          cnt++;
        }
      }
      //dist += dist * cnt / (float)attribute_number_;
      dist += cnt * weight_search;

      // float d = distance_ -> compare(x, query, (unsigned)dimension_);
      // std::cout << d << std::endl;
      dist_cout++;
      retset[i] = Neighbor(id, dist, true);
      flags[id] = true;
      L++;
    }
    // std::cout<<L<<std::endl;

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len + attribute_len, _MM_HINT_T0);
        unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len + attribute_len);
        neighbors++;
        unsigned MaxM = *neighbors;
        neighbors++;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m)
        {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          float *data = (float *)(opt_graph_ + node_size * id);
          float norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned)dimension_);

          float cnt = 0;
          char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
          for (int k = 0; k < attribute_number_; k++)
          {
            if (id_attribute[k] != attribute[k])
            {
              cnt++;
            }
          }
          //dist += dist * cnt / (float)attribute_number_;
          dist += cnt * weight_search;

          dist_cout++;
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    // for (size_t i = 0; i < L; i++) {  
    //   retset[i].flag = true;
    // }
    k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (!retset[k].flag)
      {
        retset[k].flag = true;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len + attribute_len, _MM_HINT_T0);
        unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len + attribute_len);
        unsigned MaxM = *neighbors;
        neighbors += 2;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m)
        {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          float *data = (float *)(opt_graph_ + node_size * id);
          float norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned)dimension_);

          float cnt = 0;
          char *id_attribute = (char *)(opt_graph_ + node_size * id + data_len);
          for (int k = 0; k < attribute_number_; k++)
          {
            if (id_attribute[k] != attribute[k])
            {
              cnt++;
            }
          }
          //dist += dist * cnt / (float)attribute_number_;
          dist += cnt * weight_search;

          dist_cout++;
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, false);
          int r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++)
    {
      indices[i] = retset[i].id;
    }
  }

  void IndexGraph::OptimizeGraph(float *data)
  { // use after build or load

    data_ = data;
    data_len = (dimension_ + 1) * sizeof(float);
    attribute_len = attribute_number_ * sizeof(char);
    neighbor_len = (width + 2) * sizeof(unsigned);
    node_size = data_len + attribute_len + neighbor_len;
    opt_graph_ = (char *)malloc(node_size * nd_);
    DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
    for (unsigned i = 0; i < nd_; i++)
    {
      char *cur_node_offset = opt_graph_ + i * node_size;
      float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                  data_len - sizeof(float));

      cur_node_offset += data_len;
      std::memcpy(cur_node_offset, attributes_[i].data(), attribute_len);
      cur_node_offset += attribute_len;
      unsigned k = final_graph_[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      unsigned kk = k / 2;
      std::memcpy(cur_node_offset + sizeof(unsigned), &kk, sizeof(unsigned));
      std::memcpy(cur_node_offset + 2 * sizeof(unsigned), final_graph_[i].data(),
                  k * sizeof(unsigned));
      std::vector<char>().swap(attributes_[i]);
      std::vector<unsigned>().swap(final_graph_[i]);
    }
    //free(data);
    data_ = nullptr;
    std::vector<std::vector<char>>().swap(attributes_);
    CompactGraph().swap(final_graph_);
  }

  void IndexGraph::parallel_graph_insert(unsigned id, Neighbor nn, LockGraph &g, size_t K)
  {
    LockGuard guard(g[id].lock);
    size_t l = g[id].pool.size();
    if (l == 0)
      g[id].pool.push_back(nn);
    else
    {
      g[id].pool.resize(l + 1);
      g[id].pool.reserve(l + 1);
      InsertIntoPool(g[id].pool.data(), (unsigned)l, nn);
      if (g[id].pool.size() > K)
        g[id].pool.reserve(K);
    }
  }

  void IndexGraph::GraphAdd(const float *data, unsigned n_new, unsigned dim, const Parameters &parameters)
  {
    data_ = data;
    data += nd_ * dimension_;
    assert(final_graph_.size() == nd_);
    assert(dim == dimension_);
    unsigned total = n_new + (unsigned)nd_;
    LockGraph graph_tmp(total);
    size_t K = final_graph_[0].size();
    compact_to_Lockgraph(graph_tmp);
    unsigned seed = 19930808;
#pragma omp parallel
    {
      std::mt19937 rng(seed ^ omp_get_thread_num());
#pragma omp for
      for (unsigned i = 0; i < n_new; i++)
      {
        std::vector<Neighbor> res;
        get_neighbor_to_add(data + i * dim, parameters, graph_tmp, rng, res, n_new);

        for (unsigned j = 0; j < K; j++)
        {
          parallel_graph_insert(i + (unsigned)nd_, res[j], graph_tmp, K);
          parallel_graph_insert(res[j].id, Neighbor(i + (unsigned)nd_, res[j].distance, true), graph_tmp, K);
        }
      }
    };

    std::cout << "complete: " << std::endl;
    nd_ = total;
    final_graph_.resize(total);
    for (unsigned i = 0; i < total; i++)
    {
      for (unsigned m = 0; m < K; m++)
      {
        final_graph_[i].push_back(graph_tmp[i].pool[m].id);
      }
    }
  }

  void IndexGraph::get_neighbor_to_add(const float *point,
                                       const Parameters &parameters,
                                       LockGraph &g,
                                       std::mt19937 &rng,
                                       std::vector<Neighbor> &retset,
                                       unsigned n_new)
  {
    const unsigned L = parameters.Get<unsigned>("L_ADD");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    GenRandom(rng, init_ids.data(), L / 2, n_new);
    for (unsigned i = 0; i < L / 2; i++)
      init_ids[i] += nd_;

    GenRandom(rng, init_ids.data() + L / 2, L - L / 2, (unsigned)nd_);

    unsigned n_total = (unsigned)nd_ + n_new;
    std::vector<char> flags(n_new + n_total);
    memset(flags.data(), 0, n_total * sizeof(char));
    for (unsigned i = 0; i < L; i++)
    {
      unsigned id = init_ids[i];
      float dist = distance_->compare(data_ + dimension_ * id, point, (unsigned)dimension_);
      retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;

      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        LockGuard guard(g[n].lock); //lock start
        for (unsigned m = 0; m < g[n].pool.size(); ++m)
        {
          unsigned id = g[n].pool[m].id;
          if (flags[id])
            continue;
          flags[id] = 1;
          float dist = distance_->compare(point, data_ + dimension_ * id, (unsigned)dimension_);
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          //if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
        //lock to here
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
  }

  void IndexGraph::compact_to_Lockgraph(LockGraph &g)
  {

    //g.resize(final_graph_.size());
    for (unsigned i = 0; i < final_graph_.size(); i++)
    {
      g[i].pool.reserve(final_graph_[i].size() + 1);
      for (unsigned j = 0; j < final_graph_[i].size(); j++)
      {
        float dist = distance_->compare(data_ + i * dimension_,
                                        data_ + final_graph_[i][j] * dimension_, (unsigned)dimension_);
        g[i].pool.push_back(Neighbor(final_graph_[i][j], dist, true));
      }
      std::vector<unsigned>().swap(final_graph_[i]);
    }
    CompactGraph().swap(final_graph_);
  }

  std::vector<char> IndexGraph::Attribute2int(std::vector<std::string> str)
  {
    std::vector<char> tmp;
    if (str.size() != attribute_number_)
      return tmp;
    for (int i = 0; i < str.size(); i++)
    {
      for (int j = 0; j < attributes_code[i].size(); j++)
      {
        if (str[i] == attributes_code[i][j])
          tmp.push_back(j);
      }
    }
    return tmp;
  }

  void IndexGraph::AddAllNodeAttributes(std::vector<std::string> attributes)
  {
    if (attribute_number_ != attributes.size())
    {
      attribute_number_ = attributes.size();
      std::cout << "attribute number changed to " << attribute_number_ << std::endl;
    }
    if (attributes_code.size() != attribute_number_)
    {
      attributes_code.resize(attribute_number_);
    }

    //int s = attributes_.size();
    std::vector<char> tmp;

    for (int i = 0; i < attributes.size(); i++)
    {
      int flag = 1;
      for (int j = 0; j < attributes_code[i].size(); j++)
      {
        if (attributes[i] == attributes_code[i][j])
        {
          tmp.push_back(j);
          flag--;
          break;
        }
      }
      if (flag)
      {
        tmp.push_back(attributes_code[i].size());
        attributes_code[i].push_back(attributes[i]);
      }
    }
    //attributes_[s] = tmp;
    attributes_.push_back(tmp);
  }

  bool IndexGraph::SaveAttributeTable(const std::string &fname) const
  {
    std::ofstream b_stream(fname.c_str());
    if (b_stream)
    {
      for (int i = 0; i < attribute_number_; i++)
      {
        for (int j = 0; j < attributes_code[i].size(); j++)
        {
          b_stream << attributes_code[i][j] << " ";
        }
        b_stream << std::endl;
      }

      return (b_stream.good());
    }
    else
    {
      throw std::runtime_error("[Error] Failed to save table to file: " + fname);
    }
    return false;
  }

  bool IndexGraph::LoadAttributeTable(const std::string &fname)
  {
    std::ifstream in;
    in.open(fname);
    if (in.is_open())
    {
      std::vector<std::vector<std::string>> v(attribute_number_);
      std::string s;
      const std::string c = " ";
      int query_attributes_count = 0;
      while (getline(in, s))
      {
        //const char *a = s.c_str();
        std::string::size_type pos1, pos2;
        pos2 = s.find(c);
        pos1 = 0;
        while (std::string::npos != pos2)
        {
          v[query_attributes_count].emplace_back(s.substr(pos1, pos2 - pos1));
          pos1 = pos2 + c.size();
          pos2 = s.find(c, pos1);
        }
        if (pos1 != s.length())
          v[query_attributes_count].emplace_back(s.substr(pos1));
        query_attributes_count++;
      }
      for (int i = 0; i < attribute_number_; i++)
      {
        attributes_code.emplace_back(v[i]);
      }
      in.close();
    }
    else
    {
      throw std::runtime_error("[Error] Failed to load table to file: " + fname + " not found!");
    }

    for (int i = 0; i < attributes_code.size(); i++)
    {
      for (int j = 0; j < attributes_code[i].size(); j++)
      {
        std::cout << attributes_code[i][j] << "|";
      }
      std::cout << std::endl;
    }
    return true;
  }

  void IndexGraph::fusion_distance(float &dist, float &cnt)
  {
    // dist = cnt * dist * 2 / (cnt + dist); //w_x=cnt/(dist + cnt), w_y=dist/(dist + cnt)
    // cnt *= 100; // w_x=cnt*/(dist + cnt*), w_y=dist/(dist + cnt*)
    // if(cnt == 0) cnt = 1;
    // dist = cnt * dist * 2/(cnt + dist);
    // dist = dist / 521675 + float(cnt) / 3.0;  // w_x=1/dist_max, w_y=1/cnt_max
    dist += dist * cnt / (float)attribute_number_;  //w_x=1, w_y=dist/cnt_max
    // dist += 10000 * cnt; //w_x/w_y=c, and c is a constant
  }
}

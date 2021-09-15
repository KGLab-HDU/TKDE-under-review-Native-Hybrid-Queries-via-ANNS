# Navigable Proximity Graph-driven Native Hybrid Query for Structured and Unstructured Data

## 1. Introduction

Hybrid query is an advance query processing that adds label constraint on the basis of vector similarity search; this pressing need is fueled by massive amount of emerging applications such as visual product search, academic expert finding, movie recommendation, etc. Our paper entitled "Naviagable Proximity Graph-driven Native Hybrid Query for Structured and Unstructured Data" povides a Native Hybrid Query (NHQ) solution outperforms the state-of-the-art competitors (10x faster under the same recall) on all experimental datasets.

This repo contains the code, dataset, optimal parameters, and other detailed information used for the experiments of our paper.

## 2. Compared methods

The compared methods include two categories: one is to verify the effectiveness of the proposed edge selection and routing strategies, we deploy them into NSW and KGraph to form two NPG algorithms, i.e., NPG_nsw and NPG_kgraph; the other is to evaluate the proposed hybrid query methods working off NHQ and NPG. 

**PG algorithms**

* *HNSW* ([TPAMI'2018](https://ieeexplore.ieee.org/abstract/document/8594636)) is a hierarchical PG algorihm, which is widely used in various fields and produces many optimized versions based on [hardware](https://proceedings.neurips.cc/paper/2020/hash/788d986905533aba051261497ecffcbb-Abstract.html) or [machine learning](https://dl.acm.org/doi/10.1145/3318464.3380600).
* *NSW* ([IS'2014](https://www.sciencedirect.com/science/article/abs/pii/S0306437913001300)) is the precursor of HNSW, and is a single-layer PG constructed by incrementally inserting data.
* *KGraph* ([WWW'2011](https://dl.acm.org/doi/abs/10.1145/1963405.1963487)) is an approximate K-nearest neighbor graph quickly built via NN-Descent. It only considers the distance factor when selecting edges.
* *DPG* ([TKDE'2019](https://ieeexplore.ieee.org/abstract/document/8681160)) maximizes the angle between neighbors on the basis of KGraph to alleviate redundant calculation.
* *NSG* ([VLDB'2019](http://www.vldb.org/pvldb/vol12/p461-fu.pdf)) ensures that the monotonicity of the search path by approximating the monotonic search network, thereby avoiding detouring.
* *NSSG* ([TPAMI'2021](https://ieeexplore.ieee.org/abstract/document/9383170)) is similar to DPG, it adjusts the angle between neighbors to adapt to different data characteristics, to achieve an optimal trade-off between detours and shortcuts.
* *NPG_nsw* and *NPG_kgraph* are two navigable PG (NPG) algorithms proposed by our paper.

**Hybrid query methods**

* *ADBV* ([VLDB'2020](https://dl.acm.org/doi/10.14778/3415478.3415541)) is a cost-based hybrid query method proposed by Alibaba. It implements PQ and linear scan for vector search, thus forming four sub-plans; and the least cost one is selected for hybrid query.
* *Milvus* ([SIGMOD'2021](https://dl.acm.org/doi/10.1145/3448016.3457550)) adopts a partition-based approach regarding label; it divides the object dataset through frequently used labels, and deploys ADBV on each subset.
* *Vearch* ([Middleware'2018](https://dl.acm.org/doi/10.1145/3284028.3284030)) is a repo developed by Jingdong. It first recalls similar candidates with unstructured constraint on HNSW, and then performs label filtering on the candidates to yield the final results.
* *NGT* ([SISAP'2016](https://link.springer.com/chapter/10.1007/978-3-319-46759-7_2)) is a library for performing high-speed ANNS released by Yahoo Japan. We implemented the hybrid query that conducts structured data filtering atop the candidates recalled by NGT.
* *Faiss* (IVFPQ, [TPAMI'2011](https://ieeexplore.ieee.org/abstract/document/5432202)) is popular quantization-based vector search library developed by Facebook. We deploy its hybrid query based on IVFPQ and strategy A (Fig2 in our paper).
* *SPTAG* ([ACM MM'2012](https://dl.acm.org/doi/abs/10.1145/2393347.2393378), [CVPR'2012](https://ieeexplore.ieee.org/abstract/document/6247790), [TPAMI'2014](https://ieeexplore.ieee.org/abstract/document/6549106)) is a PG-based vector similarity search library released by Microsoft, and its hybrid query works on  strategy B (Fig2 in our paper).
* *NHQ-NPG_nsw* and *NHQ-NPG_kgraph* is our hybrid query methods based on NHQ framework and two NPG algorithms.

## 3. Datasets



## 4. Parameters



## 5. Usage



## 6. Acknowledgements






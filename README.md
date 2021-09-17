# Navigable Proximity Graph-driven Native Hybrid Query for Structured and Unstructured Data

## 1. Introduction

Hybrid query is an advance query processing that adds label constraint on the basis of vector similarity search [[SIGMOD'20](https://dl.acm.org/doi/abs/10.1145/3318464.3386131), [VLDB'20](https://dl.acm.org/doi/10.14778/3415478.3415541), [SIGMOD'21](https://dl.acm.org/doi/10.1145/3448016.3457550), [KDD'21](https://dl.acm.org/doi/abs/10.1145/3447548.3470811)]; this pressing need is fueled by massive amount of emerging applications such as visual product search [[VLDB'20](https://dl.acm.org/doi/10.14778/3415478.3415541), [SIGMOD'21](https://dl.acm.org/doi/10.1145/3448016.3457550), [Middleware'18](https://dl.acm.org/doi/10.1145/3284028.3284030)], academic expert finding, movie recommendation, etc. Our paper entitled "Naviagable Proximity Graph-driven Native Hybrid Query for Structured and Unstructured Data" povides a Native Hybrid Query (NHQ) solution outperforms the state-of-the-art competitors (10x faster under the same recall) on all experimental datasets.

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

Our experiment involves eight publicly available real-world datasets and one in-house dataset. Among them, the eight public datasets are composed of high-dimensional feature vectors extracted from the unstructured data, and they do not originally contain structured labels; at this time, they are used for the performance evaluation of PG algorithms. There is no publicly available dataset thus far that contains both structured and unstructured data [[VLDB'20](https://dl.acm.org/doi/10.14778/3415478.3415541)]. Therefore, we generate corresponding label combinations for each object in public datasets following [[SIGMOD'21](https://dl.acm.org/doi/10.1145/3448016.3457550)]. For example, we add labels such as <u>*date*</u>, <u>*location*</u>, <u>*size*</u>, etc. to each image on SIFT1M to form an object dataset with structured and unstructured parts. For the in-house dataset, each object in it consisting of high-dimensional vector extracted from paper content as well as three structured attributes, i.e., <u>*affiliation*</u>, <u>*topic*</u>, <u>*publication*</u>. The following table summarizes their main information.

|           | base_num | base_dim | query_num | type        | download(vector)                                             | download (label)       |
| --------- | -------- | -------- | --------- | ----------- | ------------------------------------------------------------ | ---------------------- |
| Sift1M    | 1000000  | 128      | 10000     | Image+Label | [sift.tar.gz](http://corpus-texmex.irisa.fr/)(161MB)         | sift_label.tar.gz      |
| Gist      | 1000000  | 960      | 1000      | Image+Label | [gist.tar.gz](http://corpus-texmex.irisa.fr/)(2.6GB)         | gist_label.tar.gz      |
| Glove-100 | 1183514  | 100      | 10000     | Text+Label  | [glove-100.tar.gz](http://downloads.zjulearning.org.cn/data/glove-100.tar.gz)(424MB) | glove-100_label.tar.gz |
| Crawl     | 1989995  | 300      | 10000     | Text+Label  | [crawl.tar.gz](http://downloads.zjulearning.org.cn/data/crawl.tar.gz)(1.7GB) | crawl_label.tar.gz     |
| Audio     | 53387    | 192      | 200       | Audio+Label | [audio.tar.gz](https://drive.google.com/file/d/1fJvLMXZ8_rTrnzivvOXiy_iP91vDyQhs/view)(26MB) | audio_label.tar.gz     |
| Msong     | 992272   | 420      | 200       | Audio+Label | [msong.tar.gz](https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view)(1.4GB) | msong_label.tar.gz     |
| Enron     | 94987    | 1369     | 200       | Text+Label  | [enron.tar.gz](https://drive.google.com/file/d/1TqV43kzuNYgAYXvXTKsAG1-ZKtcaYsmr/view)(51MB) | enron_label.tar.gz     |
| UQ-V      | 1000000  | 256      | 10000     | Video+Label | [uqv.tar.gz](https://drive.google.com/file/d/1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF/view?usp=sharing)(800MB) | uqv_label.tar.gz       |
| Paper     | 2029997  | 200      | 10000     | Text+Label  | [paper.tar.gz](https://drive.google.com/file/d/1t4b93_1Viuudzd5D3I6_9_9Guwm1vmTn/view)(1.41GB) | paper_label.tar.gz     |

Note that, all base data and query data are converted to `fvecs` format, and groundtruth data is converted to `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for the description of `fvecs` and `ivecs` format.

## 4. Parameters

Because parameters' adjustment in the entire base dataset may cause overfitting, we randomly sample a certain percentage of data points from the base dataset to form a validation dataset. We search for the optimal value of all the adjustable parameters of each algorithm on each validation dataset, to make the algorithms' search performance reach the optimal level. See the [parameters](parameters) page for more information.

## 5. Installation and Usage

**Prerequistes**

```
GCC 4.9+ wih OpenMP
CMake 2.8+
Boost 1.55+
```

**Installation**

You can run the `build.sh` script to install all algorithms, including NPG_kgraph, NPG_nsw, NHQ-NPG_kgraph and NHQ-NPG_nsw.

**Usage**

After performing the installation, you can test each algorithm via the script `test_hybrid_query.py` or `test_npg.py`, and the specific parameter values can be found in [parameters](parameters).

**Validation of NHQ framework**

To verify the capabilities of NHQ, we additionally implement the hybrid query based on the first vector search and then label filtering across NPG_kgraph and NPG_nsw. Thus, for hybrid query on NHQ, you can test it by setting the following options in the script `test_hybrid_query.py`:

```
<index>: 'NHQ-NPG_kgraph' or 'NHQ-NPG_nsw'	# build and search
```

For the first vector search then label filtering, you can test it by setting the following options in `test_hybrid_query.py`:

```
<index>: 'NPG_kgraph' or 'NPG_nsw'	# build and search
```

**Performance of NPG**

To verify the effectiveness of the proposed edge selection and routing strategies, you can evaluate NPG algorithms (i.e., NPG_kgraph and NPG_nsw) by setting the following options in `test_npg.py`:

```
<index>: 'NPG_kgraph' or 'NPG_nsw'	# build and search
```

**Evaluation of Hybrid Query Methods**

You can test it by setting the following options in the script `test_hybrid_query.py`:

```
<index>: 'NHQ-NPG_kgraph' or 'NHQ-NPG_nsw'	# build and search
```

**Parameter Sensitivity**

For NHQ, $\omega _{x}$​ and $\omega _{y}$​ in Equation 5 of our paper are a pair of parameters that regulate the weights of $\delta$​ and $\chi$​, where $\delta$​ is the distance between unstructured data, and $\chi$​ is the distance between structured data. Thus, $\omega _{x}$​ and $\omega _{y}$​ will impact the hybrid query performance. You can set different $\omega _{x}$​ and $\omega _{y}$​​ by modify the following code in `/NHQ-NPG_kgraph/src/index_graph.cpp`.

```c++
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
```

## 6. Acknowledgements

The implementation of NPG_kgraph is taken from [kgraph](https://github.com/aaalgo/kgraph), [ssg](https://github.com/ZJULearning/ssg), [nns_benchmark](https://github.com/DBAIWangGroup/nns_benchmark/tree/master/algorithms/DPG), and the implementation of NPG_nsw is taken from [n2](https://github.com/kakao/n2). Many thanks to them for inspiration. Thanks to everyone who provided references for this project.

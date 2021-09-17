# NHQ-NPG_kgraph

## Compile on Linux

Go to the root directory of NHQ-NPG_kgraph and run the following scripts.    

```shell
cd NHQ-NPG_kgraph/
mkdir build && cd build
cmake ..
make
```

## Build NHQ-NPG_kgraph index
First: 

```shell
cd NHQ-NPG_kgraph/build/tests/
```

Then: 

```shell
./test_dng_index data_file att_file save_graph save_attributetable K L iter S R Range PL B M
```

 Meaning of the parameters:    

```
<data_file> is the path of the origin data.
<att_file> is the path of the corresponding structured labels of the origin data.
<save_graph> is the path of the NHQ-NPG_kgraph to be saved.
<save_attributetable> is the path of the attributes codes to be saved.
<K> is the 'K' of kNN graph.
<L> is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
<iter> is the parameter controlling the maximum iteration times, iter usually < 30.
<S> is the parameter contollling the graph quality, larger is more accurate but slower.
<R> is the parameter controlling the graph quality, larger is more accurate but slower.
<RANGE> controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
<PL> controls the quality of the NHQ-NPG_kgraph, the larger the better.
<B> controls the quality of the NHQ-NPG_kgraph.
<M> control the edge selection of NHQ-NPG_kgraph.
```

## Search on NHQ-NPG_kgraph
```shell
./test_dng_optimized_search graph_path attributetable_path data_path query_path query_att_path groundtruth_path
```

 Meaning of the parameters:    

```
<graph_path> is the path of the pre-built NHQ-NPG_kgraph.
<attributetable_path> is the path of the attributes codes.
<data_path> is the path of the origin data.
<query_path> is the path of the query data.
<query_att_path> is the path of the corresponding structured labels of the query data.
<groundtruth_path> is the path of the groundtruth data.
```

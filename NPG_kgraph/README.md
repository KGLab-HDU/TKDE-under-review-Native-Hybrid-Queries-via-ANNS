# NPG_kgraph

## Compile on Linux

Go to the root directory of NPG_kgraph and run the following scripts.    

```shell
cd NPG_kgraph/
mkdir build && cd build
cmake ..
make
```

## Build NPG_kgraph index
First: 

```shell
cd NPG_kgraph/build/tests/
```

Then: 

```shell
./test_dng_index data_file save_graph K L iter S R RANGE PL B M
```

Meaning of the parameters:    

```
<data_file> is the path of the original object set.
<save_graph> is the path of the NPG_kgraph to be saved.
<K> is the 'K' of kNN graph.
<L> is the parameter controlling the graph quality, larger is more accurate but slower, no smaller than K.
<iter> is the parameter controlling the maximum iteration times, iter usually < 30.
<S> is the parameter contolling the graph quality, larger is more accurate but slower.
<R> is the parameter controlling the graph quality, larger is more accurate but slower.
<RANGE> controls the index size of the graph, the best R is related to the intrinsic dimension of the dataset.
<PL> controls the quality of the NPG_kgraph, the larger the better.
<B> controls the quality of the NPG_kgraph.
<M> control the edge selection of NPG_kgraph.
```

### Vector Search on NPG_kgraph
```shell
./test_dng_optimized_search graph_path data_path query_path groundtruth_path
```

Meaning of the parameters:    

```
<graph_path> is the path of the pre-built NPG_kgraph.
<data_path> is the path of the original object set.
<query_path> is the path of the query object.
<groundtruth_path> is the path of the groundtruth data.
```
### Hybrid Queries on NPG_kgraph ("first vector similarity search, then attribute filtering")
```shell
./test_dng_hybrid_search graph_path data_path query_path base_att_path query_att_path groundtruth_path
```

Meaning of the parameters:  

```
<graph_path> is the path of the pre-built NPG_kgraph.
<data_path> is the path of the original object set.
<query_path> is the path of the query object.
<base_att_path> is the path of the corresponding structured attributes of the original ojects.
<query_att_path> is the path of the corresponding structured attributes of the query object.
<groundtruth_path> is the path of the groundtruth data.
```
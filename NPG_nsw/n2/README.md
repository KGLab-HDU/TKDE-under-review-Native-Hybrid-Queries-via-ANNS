# NPG_nsw

## Compile on Linux  

```shell
cd NPG_nsw/n2/
make
cd examples/cpp/
make
```

## Build NPG_nsw Index
First: 

```shell
cd NPG_nsw/n2/examples/cpp/
```

Then: 

```shell
./index data_file save_graph Max_M0 efConstruction
```

Meaning of the parameters:    

```
<data_file> is the path of the origin data.
<save_graph> is the path of the NPG_nsw to be saved.
<Max_M0> is the 'K' of kNN graph.
<efConstruction> is the parameter contollling the graph quality, larger is more accurate but slower.
```

### Search on NPG_nsw
```shell
./search graph_path query_file groundtruth_file
```

Meaning of the parameters:    

```
<graph_path> is the path of the pre-built NPG_nsw.
<query_file> is the path of the query data.
<groundtruth_file> is the path of the groundtruth data.
```
### Hybrid Search on NPG_nsw (first vector search then label filtering)
```shell
./hybrid_search graph_path data_path query_path base_att_path query_att_path groundtruth_path
```

Meaning of the parameters:  

```
<graph_path> is the path of the pre-built NPG_nsw.
<data_path> is the path of the origin data.
<query_path> is the path of the query data.
<base_att_path> is the path of the corresponding structured labels of the origin data.
<query_att_path> is the path of the corresponding structured labels of the query data.
<groundtruth_path> is the path of the groundtruth data.
```


# NPG_nsw

## Build  

```shell
cd NPG_nsw/n2/
make
cd examples/cpp/
make
```

## How To Use

### Build Index 
First: 

```shell
cd NPG_nsw/n2/examples/cpp/
```

Then: 

```shell
./index data_file save_graph max_m0 ef_construction
```

 Meaning of the parameters:    

    **data_file** is the path of the origin data.
    **save_graph** is the path of the DNG to be saved.
    **max_m0** is the 'K' of kNN graph.
    **ef_construction** is the parameter contollling the graph quality, larger is more accurate but slower.

### Search
```shell
./search graph_path query_file groundtruth_file
```

 Meaning of the parameters:    

```
graph_path is the path of the pre-built DNG.
query_file is the path of the query data.
groundtruth_file is the path of the groundtruth data.
```
### hybrid_Search
```shell
./hybrid_search graph_path data_path query_path base_att_path query_att_path groundtruth_path
```

 Meaning of the parameters:  

```
graph_path is the path of the pre-built DNG.
data_path is the path of the origin data.
query_path is the path of the query data.
base_att_path is the path of the corresponding structured labels of the origin data.
query_att_path is the path of the corresponding structured labels of the query data.
groundtruth_path is the path of the groundtruth data.
```



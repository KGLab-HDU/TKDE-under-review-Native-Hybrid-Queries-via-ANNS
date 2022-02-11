# NHQ-NPG_nsw

## Compile on Linux

```shell
cd NHQ-NPG_nsw/
make
cd examples/cpp/
make
```

## Build NHQ-NPG_nsw Index 
First: 

```shell
cd NHQ-NPG_nsw/examples/cpp/
```

Then: 

```shell
./index data_file att_file save_graph save_attributetable MaxM0 efConstruction
```

Meaning of the parameters:    

```
<data_file> is the path of the original object set.
<att_file> is the path of the corresponding structured attributes of the original objects.
<save_graph> is the path of the NPG_kgraph to be saved.
<save_attributetable> is the path of the attributes codes to be saved.
<MaxM0> is the 'K' of kNN graph.
<efConstruction> is the parameter contollling the graph quality, larger is more accurate but slower.
```

## Search on NHQ-NPG_nsw
```shell
./search graph_file attributetable_file query_file groundtruth_file attributes_query_file
```

Meaning of the parameters:    

```
<graph_file> is the path of the pre-built NPG_kgraph.
<attributetable_file> is the path of the attributes codes.
<query_file> is the path of the query object.
<groundtruth_file> is the path of the groundtruth data.
<attributes_query_file> is the path of the corresponding structured attributes of the query object.
```

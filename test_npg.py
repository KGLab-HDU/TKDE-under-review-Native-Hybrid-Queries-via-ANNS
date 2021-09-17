import os
import sys

if(len(sys.argv) != 2):
    print(sys.argv[0]+' build_or_search')
    sys.exit(1)

bos = sys.argv[1]

if(bos == 'build'):
    index = input(
        "Choose an index (NPG_kgraph or NPG_nsw): ")
    if (index == 'NPG_kgraph'):
        data_path = input("Input origin data path: ")
        save_graph = input("Input path to save the index: ")
        K = input("index parameter K: ")
        L = input("index parameter L: ")
        iter = input("index parameter iter: ")
        S = input("index parameter S: ")
        R = input("index parameter R: ")
        Range = input("index parameter Range: ")
        PL = input("index parameter PL: ")
        B = input("index parameter B: ")
        M = input("index parameter M: ")
        os.system("./NPG_kgraph/build/tests/test_dng_index %s %s %s %s %s %s %s %s %s %s" %
                  (data_path, save_graph, K, L, iter, S, R, Range, PL, B, M))
    elif (index == 'NPG_nsw'):
        data_path = input("Input origin data path: ")
        save_graph = input("Input path to save the index: ")
        m = input("index parameter maxm0: ")
        efc = input("index parameter efconstruction: ")
        os.system("./NPG_nsw/n2/examples/cpp/index %s %s %s %s" %
                  (data_path, save_graph, m, efc))
    else:
        print('Invalid Index')
        sys.exit(1)
elif(bos == 'search'):
    index = input(
        "Choose an index (NPG_kgraph or NPG_nsw): ")
    if (index == 'NPG_kgraph'):
        graph_path = input("Input index path: ")
        data_path = input("Input origin data path: ")
        query_path = input("Input query data path: ")
        ground_path = input("Input groundtruth data path:")
        os.system("./NPG_kgraph/build/tests/test_dng_optimized_search %s %s %s %s %s %s" %
                  (graph_path, data_path, query_path, ground_path))
    elif (index == 'NPG_nsw'):
        graph_path = input("Input index path: ")
        data_path = input("Input origin data path: ")
        query_path = input("Input query data path: ")
        ground_path = input("Input groundtruth data path:")
        os.system("./NPG_nsw/n2/examples/cpp/search %s %s %s %s %s %s" %
                  (graph_path, data_path, query_path, ground_path))
    else:
        print('Invalid Index')
        sys.exit(1)
else:
    print('Invalid Operate')

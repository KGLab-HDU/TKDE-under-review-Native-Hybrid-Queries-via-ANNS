# Parameters

The following tables show the optimal parameters obtained by grid search on validation dataset.

## NHQ-NPG_kgraph

Please see [here](../NHQ-NPG_kgraph/) for more detailed meaning about these parameters.

|       | SIFT1M | GIST1M | GloVe | Crawl | Audio | Msong | Enron | UQ-V | Paper |
| :---: | :----: | :----: | :---: | :---: | :---: | :---: | :---: | :--: | :---: |
|   K   |  100   |  100   |  100  |  100  |  100  |  400  |  100  | 100  |  100  |
|   L   |  100   |  100   |  100  |  100  |  130  |  420  |  100  | 100  |  100  |
| iter  |   12   |   12   |  12   |  12   |  12   |  12   |  12   |  12  |  12   |
|   S   |   10   |   10   |  10   |  10   |  10   |  10   |  10   |  15  |  10   |
|   R   |  300   |  300   |  100  |  100  |  100  |  300  |  100  | 200  |  100  |
| RANGE |   20   |   20   |  20   |  20   |  20   |  80   |  20   |  20  |  20   |
|  PL   |  350   |   50   |  150  |  150  |  250  |  150  |  50   | 150  |  50   |
|   B   |  0.4   |  0.6   |  0.2  |  0.4  |  0.6  |  0.6  |  0.2  | 0.6  |  0.2  |
|   M   |  1.0   |  1.0   |  1.0  |  1.0  |  1.0  |  1.0  |  1.0  | 1.0  |  1.0  |

## NHQ-NPG_nsw

Please see [here](../NHQ-NPG_nsw/) for more detailed meaning about these parameters.

|                | SIFT1M | GIST1M | GloVe | Crawl | Audio | Msong | Enron | UQ-V | Paper |
| :------------: | :----: | :----: | :---: | :---: | :---: | :---: | :---: | :--: | :---: |
|     maxm0      |   40   |   20   |  20   |  100  |  60   |  50   |  90   |  30  |  90   |
| efconstruction |  100   |  200   |  900  |  500  |  800  |  900  |  900  | 300  |  400  |

## NPG_kgraph

Please see [here](../NPG_kgraph/) for more detailed meaning about these parameters.

|       | SIFT1M | GIST1M | GloVe | Crawl |
| :---: | :----: | :----: | :---: | :---: |
|   K   |  200   |  100   |  400  |  400  |
| L_add |  220   |  130   |  420  |  400  |
|   S   |   25   |   20   |  25   |  20   |
| R_KG  |  200   |  300   |  300  |  200  |
| RANGE |   40   |   40   |  90   |  90   |
|  PL   |   50   |  350   |  50   |  350  |
|   B   |  1.0   |  1.0   |  0.8  |  1.0  |
|   M   |  1.0   |  1.0   |  1.0  |  1.0  |

## NPG_nsw

Please see [here](../NPG_nsw/n2/) for more detailed meaning about these parameters.

|    dataset     | SIFT1M | GIST1M | GloVe | Crawl |
| :------------: | :----: | :----: | :---: | :---: |
|     maxm0      |   40   |   60   |  60   |  60   |
| efconstruction |  300   |  300   |  700  |  300  |

## Weight_search

When performing hybrid query, we set $\omega _{x}=1$ and $\omega _{y}=$weight_search as follows.

|               | SIFT1M | GIST1M | GloVe | Crawl | Audio | Msong | Enron  | UQ-V | Paper |
| :-----------: | :----: | :----: | :---: | :---: | :---: | :---: | :----: | :--: | :---: |
| weight_search | 140000 |   9    |  80   |  90   | 3e+10 | 5500  | 480000 |  2   | 5000  |
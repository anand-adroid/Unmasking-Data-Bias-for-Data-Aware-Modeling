# Unmasking-Data-Bias-for-Data-Aware-Modeling

Healthcare machine learning (HML) models rely on clinical research databases (CRDs) like Medical Information Mart for Intensive Care (MIMIC), which inherently exhibits bias within the healthcare system. While algorithmic bias has received considerable attention, the data bias in HML models remains understudied.
Our systematic survey conducted over half a decade of HML research reveals a concerning trend: bias in ethnic demographic variables consideration during
model training. Our data analysis shows a noteworthy association between ethnic demographics and task-specific predictions, emphasizing the need to consider
model performance across all demographics for fair, generalizable and data-aware HML models. Motivated from recent works of data transparency, we provide a
comprehensive datasheet for MIMIC IV v2.0 CRD to aid researchers fully comprehending underlying data, to mitigate inherent data bias and effectively utilize it for
task-specific purposes.

## Install from PyPI
You can directly install our benchmark by `pip install nas_bench_graph`

## Usage 
First, read the benchmark of a certain dataset by specifying the name. The nine supported datasets are: cora, citeseer, pubmed, cs, physics, photo, computers, arxiv, and proteins. For example, for the Cora dataset:
```
from nas_bench_graph import lightread
bench = lightread('cora')
```
The data is stored as a `dict` in Python.

Then, an architecture needs to be specified by its macro space and operations.
We consider the macro space as a directed acyclic graph (DAG) and constrain the DAG to have only one input node for each intermediate node. Therefore, the macro space can be specificed by a list of integers, indicating the input node index for each computing node (0 for the raw input, 1 for the first computing node, etc.). Then, the operations can be specified by a list of strings with the same length. For example, we provide the code to specify the architecture in the following figure:
![arch](https://user-images.githubusercontent.com/17705534/173767528-eda1bc64-f4d8-4da1-a0e9-8470f55ccc6a.png)

```
from nas_bench_graph import Arch
arch = Arch([0, 1, 2, 1], ['gcn', 'gin', 'fc', 'cheb'])
# 0 means the inital computing node is connected to the input node
# 1 means the next computing node is connected to the first computing node
# 2 means the next computing node is connected to the second computing node 
# 1 means there is another computing node connected to the first computing node
```

Notice that we assume all leaf nodes (i.e., nodes without descendants) are connected to the output, so there is no need to specific the output node. 

Besides, the list can be specified in any order, e.g., the following code can specific the same architecture:
```
arch = Arch([0, 1, 1, 2], ['gcn', 'cheb', 'gin', 'fc'])
```

The benchmark data can be obtained by a look-up table. In this repository, we only provide the validation and test performance, the latency, and the number of parameters as follows:

```
info = bench[arch.valid_hash()]
info['valid_perf']   # validation performance
info['perf']         # test performance
info['latency']      # latency
info['para']         # number of parameters
```

For the complete benchmark, please downloadfrom https://figshare.com/articles/dataset/NAS-bench-Graph/20070371, which contains the training/validation/testing performance at each epoch. Since we run each dataset with three random seeds, each dataset has 3 files, e.g.,

```
from nas_bench_graph import read
bench = read('cora0.bench')   # cora1.bench and cora2.bench 
```

The full metric for any epoch can be obtained as follows.
```
info = bench[arch.valid_hash()]
epoch = 50
info['dur'][epoch][0]   # training performance
info['dur'][epoch][1]   # validation performance
info['dur'][epoch][2]   # testing performance
info['dur'][epoch][3]   # training loss
info['dur'][epoch][4]   # validation loss
info['dur'][epoch][5]   # testing loss
info['dur'][epoch][6]   # best performance
```

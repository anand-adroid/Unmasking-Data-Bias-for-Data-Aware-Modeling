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
df = pd.DataFrame(data)
```

The data is stored as a `dict` in Python.

```
# Create a contingency table
contingency_table = pd.pivot_table(df, values='count', index=['race_sub_group', 'insurance'], columns='label', fill_value=0)
```

```
# Perform the chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)
```

```
# Print the contingency table
print("Contingency Table:")
print(contingency_table)
```

```
# Print the test statistic and p-value
print("Chi-square test statistic:", chi2)
print("p-value:", p)
```




For the complete benchmark, please downloadfrom https://figshare.com/articles/dataset/NAS-bench-Graph/20070371, which contains the training/validation/testing performance at each epoch. Since we run each dataset with three random seeds, each dataset has 3 files, e.g.,

```

```

The full metric for any epoch can be obtained as follows.


# Plausibilization of Partial INDs with ML

- To help with creating a model by which we can evaluate the validity of our detected INDs
- Train a simple ML model: Decision Tree, Random Forest, Naive Bayes, SVM, etc.

## Features

- Original Columns:
  - Distinct-Ratio
- Sampled Columns:
  - Count
  - #Distinct
  - Sampling Ratio
- Missing Values
- Missing Ratio - Highly correlated with Missing Values
- Total Missing Ratio
- Useless
- Ratio of Sample Size
- Ratio of Cardinality

From Rostin Paper:

- Coverage (?) - Number of values in B, that are in A
- DependentAndReferenced - Counts how often A appears as a right side of an IND
- MultiDependent - Count how often A appears as left side of an IND
- ColumnName Similarity - (Edit-Distance, Levenshtein Distance, Hamming Distance)
- ValueLengthDiff - Differences between average value length
- OutOfRange - Percentage of values of B, that are outside of range(A)
- TypicalNameSuffix - Checks if column name ends in substring indicating FK
- TableSizeRatio - count(A)/count(B)

## Model Selection

- Decision Tree
- Random Forest
- Naive Bayes
- SVM

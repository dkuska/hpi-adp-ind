# Plausibilisation of INDs

A_s [= B_s => A [= B

Note: We are only talking about unique values here. Both inside A,B, A_s & B_s

Note:
As an approximation we might go in 2 'steps':
A_s [= B_s => A_s [= B => A [= B

## What we know

|A|, |B| - Cardinalities of the original columns
|A_s|, |B_s| - Cardinalities of our sample

### Sampling Ratios
|A_s|/|A| = sr_A - Sampling Ratio of A
|B_s|/|B| = sr_B - Sampling Ratio of B

### Ratio of Sample Sizes
|A|/|B| = rss

### Ratio of Cardinality
|A_s|/ |B_s|

### Missing values
|A_s\B_s| = mv_(A,B)
Number of values in A, that are not in B

### Missing Ratio
|A_s\B_s|\|A_s| = mr_(A,B)
Ratio of values in A, that are not in B

### Useless Ratio
|A_s\B_s|\|B_s| = ur_(A,B)
Ratio of values in B, that are not in A.
What is 'extra' in B
i.e. they have no value for the IND A [= B

### Mirrored Missing Values
|B_s\A_s| - mv(B,A)
Missing Values of the IND B [= A

## Exclusion Criteria

- |A|/|B| > 1 => FP
- min(A) < min(B)
- max(A) > max(B)
- min(A_s) < min(B)
- max(A_s) > max(B)

## Further Plausbilisations

- If sr_B == 1 but mv = 0 => FP
- If sr_B is large, but mv/mr is also large => Unlikely TP
- If mv > |B\B_s| => FP, not enough values left in B to make IND true

## Most important Values

- sr_A, sr_B => The higher, the more confident we are
- mv, mr => The lower, the more confident we are

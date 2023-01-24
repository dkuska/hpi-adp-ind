# Recap
## Precision - TP / TP + FP
## Recall - TP / TP + FN

## For us Recall > Precision 
## ==> Try to keep all real INDs while allowing non-INDs to slip through
## FP << FN

# Baseline
## Decision Tree
Recall:            0.14263517712865134

Precision:         0.1859052247873633
## Random Forest
Recall:            0.0

Precision:         0.0
## Logistic Regression
Recall:            0.0

Precision:         0.0

# Random_OverSampling
## Decision Tree
Recall:            0.19608452454940958

Precision:         0.09680883706658484
## Random Forest
Recall:            0.8604723430702299

Precision:         0.03197385741668784
## Logistic Regression
Recall:            0.5779987569919205

Precision:         0.030017913916369446

# SMOTE
## Decision Tree
Recall:            0.1497824735860783

Precision:         0.13076505697232774
## Random Forest
Recall:            0.6718458669981355

Precision:         0.035257664709719506
## Logistic Regression
Recall:            0.5957116221255438

Precision:         0.030548340318391154

# ADASYN
## Decision Tree
Recall:            0.14263517712865134

Precision:         0.12554704595185995
## Random Forest
Recall:            0.6687383467992541

Precision:         0.03525152751158943
## Logistic Regression
Recall:            0.5870105655686761

Precision:         0.03008392922552595

# Random_UnderSampling
## Decision Tree
Recall:            0.6892479801118707

Precision:         0.04492515849385267
## Random Forest
Recall:            0.8542573026724674

Precision:         0.03202805513159581
## Logistic Regression
Recall:            0.5954008701056557

Precision:         0.03057528125748025

# After Hyperparameter Tuning, with UnderSampling:
## Decision Tree
Recall:            0.6133768352365416

Precision:         0.04366073354013569
## Random Forest
Recall:            0.7455138662316476

Precision:         0.049635376260667184
## Ridge Regression
Recall:            0.6213003961780471

Precision:         0.030586021752099492
## Logistic Regression
Recall:            0.6226986716383127

Precision:         0.03066013379384732
## SVM
Recall:            0.8217198788161267

Precision:         0.026054058847000752

# Lessons learned:

- Decision Trees suck
- Random Forests >> Decision Trees
- Feature Importance: All approx. equally important (Mean decrease in impurity), however MDI has bias towards features with different cardinalities
- Naive Bayes blows up since we don't know all possible priors
- Logistic Regression >> Ridge Regression
- SVM suprisingly good
- Moving forwards: Random Forest, Logistic Regression and SVM

# TODOS:

- Generate Data for more Sampling Ratess
- Figure out Precision/Recall tradeoff
- Normalizations/Regularisation of features shows no real difference
- Explore other UnderSampling Methods except Random
- Filter Training Data with 'hard' rules
- Train for predict_proba
- Put model in 'production'
- 1 Modell pro Sampling Methode
- Neue Trainingsdaten mit PartialSPIDER missing_values groesser, bzw. relativ

# Experimente Report:

- Budget, variieren
- Welche Metadaten wir zur Verfuegung haben, abschaetzen was das an unseren Ergebnissen aendert
- Minhash - sampling methode, k-kleinste Hash Werte

- Problem Statement:
  - Budget B, columns C
  - Minimum 4 Datensaetze
  - Random Sampling mehrmals wiederholen

- Runtime

- 

- Precision/Recall - F1-Score/ F-beta Score
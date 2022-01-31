# Handling Imbalanced Data
An imbalanced classification problem is a problem that involves predicting a class label where the distribution of class labels in the training dataset is not equal. 
Imbalanced classification is primarily challenging as a predictive modeling task because of the
severely skewed class distribution. This is the cause for poor performance with traditional
machine learning models and evaluation metrics that assume a balanced class distribution.

As a dataset, I have used Credit Card Fraud Detection
from Kaggle https://www.kaggle.com/mlg-ulb/creditcardfraud where our purpose is to determine fraudulent credit card transactions. 

__img__

# Evaluation

Firstly, because most of the standard
metrics that are widely used assume a balanced class distribution, and because typically not
all classes, and therefore, not all prediction errors, are equal for imbalanced classification. So, accuracy is inappropriate for
imbalanced classification problems. 

The main reason is that the overwhelming number of
examples from the majority class (or classes) will overwhelm the number of examples in the
minority class, meaning that even unskillful models can achieve accuracy scores of 90 percent,
or 99 percent, depending on how severe the class imbalance happens to be. 

As a evaluation metric I am using __balanced_accuracy_score, roc_auc_score, geometric_mean_score.__



## 1. Data Sampling Methods

In order to avoid these problems I have used a few approaches that work with imbalanced datasets. Data sampling provides a collection of techniques that transform a training dataset in
order to balance or better balance the class distribution. lthough often described in terms of two-class classification problems, class imbalance also
affects those datasets with more than two classes that may have multiple minority classes
or multiple majority classes. __Oversampling methods duplicate examples in the minority class or synthesize new examples
from the examples in the minority class. Meanwhile, Undersampling methods delete or select a subset of examples from the majority class.__ 

### 1.1 SMOTE (Synthetic Minority Oversampling Technique)

SMOTE first selects a minority class instance a at random and finds its k nearest
minority class neighbors. The synthetic instance is then created by choosing one of
the k nearest neighbors b at random and connecting a and b to form a line segment
in the feature space. The synthetic instances are generated as a convex combination
of the two chosen instances a and b.

### 1.2  ADASYN (Adaptive Synthetic Sampling)

It is a modification of SMOTE that is based on the idea of adaptively generating minority data samples
according to their distributions: more synthetic data is generated for minority class
samples that are harder to learn compared to those minority samples that are easier
to learn.


### 1.3 Borderline-SMOTE

A popular extension to SMOTE involves selecting those instances of the minority class that are
misclassified, such as with a k-nearest neighbor classification model. We can then oversample
just those difficult instances, providing more resolution only where it may be required.


### 1.4 SVM-SMOTE

This method is an alternative to Borderline-SMOTE where a SVM algorithm
is used instead of a KNN to identify misclassified examples on the decision boundary. In the SVMSMOTE(), borderline area is approximated by the support vectors obtained after training
a standard SVMs classifier on the original training set. New instances will be
randomly created along the lines joining each minority class support vector with a
number of its nearest neighbors using the interpolation


### 1.5 Near Miss Undersampling

In this method, we have three versions of
the technique, named NearMiss-1, NearMiss-2, and NearMiss-3. 

Here, distance is determined in feature space
using Euclidean distance or similar.

 NearMiss-1: Majority class examples with minimum average distance to three closest
minority class examples.

 NearMiss-2: Majority class examples with minimum average distance to three furthest
minority class examples.

 NearMiss-3: Majority class examples with minimum distance to each minority class
example.


# 2. Probalistic models

Probabilistic models are
those models that are fit on the data under a probabilistic framework and often perform well
in general for imbalanced classification dataset. I will evaluate a suite of models that are known to be effective at predicting probabilities.

Specifically, these are models that are fit under a probabilistic framework and explicitly predict a
calibrated probability for each example. A such, this makes them well-suited to this dataset, even
with the class imbalance. We will evaluate the following six probabilistic models implemented
with the scikit-learn library:

#### Logistic Regression (LR)
#### Linear Discriminant Analysis (LDA)
#### Quadratic Discriminant Analysis (QDA)
#### Gaussian Naive Bayes (GNB)
#### Gaussian Process (GPC)

# 3. Cost sensitive models

Some machine learning algorithms can be adapted to pay more attention to one class than
another when fitting the model. These are referred to as cost-sensitive machine learning
models and they can be used for imbalanced classification by specifying a cost that is inversely
proportional to the class distribution.

Cost-sensitive learning is a subfield of machine learning that takes the costs of prediction
errors (and potentially other costs) into account when training a machine learning model. It is
a field of study that is closely related to the field of imbalanced learning that is concerned with
classification on datasets with a skewed class distribution.


# 4. Results

__img__

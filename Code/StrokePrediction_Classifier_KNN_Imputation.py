#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, precision_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


# ### Read Dataset

# In[3]:


patients = pd.read_csv('healthcare-dataset-stroke-data.csv')
patients.sample(5)


# In[6]:


mask = (patients.stroke == 1) & ((patients.smoking_status == "formerly smoked") | (patients.smoking_status == "smokes"))
patients[mask]


# In[7]:


patients.gender.value_counts()


# ### Functions

# In[6]:


def get_k_cross_validation_metrics(clf, X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    cv_results = cross_validate(clf, X, y, cv=skf, scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'ROC': make_scorer(roc_auc_score),
    })
    return cv_results


# In[7]:


def name(cls):
    return cls.__class__.__name__

def get_accuracys(classifiers, X, y, k):
    cls_accs = {name(cls):[] for cls in classifiers}
    for cls in classifiers:
        cv_results = get_k_cross_validation_metrics(cls, X, y, k)
        cls_accs[name(cls)] = cv_results
    return cls_accs


# ### Model Training KNN Imputer)

# In[15]:


X = patients.drop(["stroke"], axis=1)
y = patients.stroke


# In[18]:


k_folds = 10
knn = KNeighborsClassifier(n_neighbors=5)
svc = svm.SVC(probability=True)
adaBoost = AdaBoostClassifier(n_estimators=100) # default n_estimators=50
classifiers = [knn, svc, adaBoost]
cv_results = get_accuracys(classifiers, X, y, k_folds)


# In[19]:


metrics = ['accuracy', 'precision', 'recall','f1', 'specificity', 'ROC']
result = pd.DataFrame([], columns=["classifier", *metrics])
for cls, cls_metrics in cv_results.items():
    scores = {metric:cls_metrics['test_' + metric].mean() for metric in metrics}
    scores["classifier"] = cls
    result = result.append(scores, ignore_index=True)
result


# In[20]:


result.to_excel("PredictResult(balanced).xlsx", sheet_name="balanced", float_format="%.4f", index=False)


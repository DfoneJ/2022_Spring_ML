#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[103]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, precision_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')


# ### Read Dataset

# In[104]:


patients = pd.read_csv('healthcare-dataset-stroke-data-preprocessed.csv')
patients[["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi", "smoking_status", "work_type_Self-employed", "stroke"]]
patients.sample(10)


# ### Functions

# In[105]:


def get_predict_after_augmentation(clf, X_train, y_train, X_test):
    adasyn = ADASYN(n_neighbors=3)
    X_balance, y_balance = adasyn.fit_resample(X_train, y_train.astype('int'))
    predict = clf.fit(X_balance, y_balance).predict(X_test)
    return predict

def get_metric_scores(predict, actual):
    scores = {}
    scores['accuracy'] = accuracy_score(actual, predict)
    scores['precision'] = precision_score(actual, predict)
    scores['recall'] = recall_score(actual, predict)
    scores['f1'] = f1_score(actual, predict)
    scores['ROC'] = roc_auc_score(actual, predict)
    return scores


# In[106]:


def build_gmms(X, y, mixiures=2):
    gmms = []
    for clazz in [0, 1]:
        gmm = GaussianMixture(n_components=mixiures)
        gmm.fit(X[y == clazz])
        gmms.append(gmm)
    return gmms


# In[107]:


def get_k_cross_validation_metrics(clf, X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf_splited = skf.split(X, y)
    
    metrics = ['accuracy', 'precision', 'recall','f1', 'ROC']
    cv_results = {k:np.array([]) for k in metrics}
    for i, (train_index, test_index) in enumerate(skf_splited):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        gmms = build_gmms(X_train, y_train)
        probs = [np.exp(gmm.score_samples(X_test)) for gmm in gmms]
        y_predict = np.argmax(probs, axis=0)
#         y_predict = get_predict_after_augmentation(clf, X_train, y_train, X_test)
#         y_predict = clf.fit(X_train, y_train).predict(X_test)
        scores = get_metric_scores(y_predict, y_test)
        
        for metric in scores.keys():
            cv_results[metric] = np.append(cv_results[metric], scores[metric])
        
    return cv_results


# In[108]:


def name(cls):
    return 'GMM'

def get_accuracys(classifiers, X, y, k):
    cls_accs = {name(cls):[] for cls in classifiers}
    for cls in classifiers:
        cv_results = get_k_cross_validation_metrics(cls, X, y, k)
        cls_accs[name(cls)] = cv_results
    return cls_accs


# In[109]:


def build_result(cv_results):
    metrics = ['accuracy', 'precision', 'recall','f1', 'ROC']
    result = pd.DataFrame([], columns=["classifier", *metrics])
    for cls, cls_metrics in cv_results.items():
        scores = {metric:cls_metrics[metric].mean() for metric in metrics}
        scores["classifier"] = cls
        result = result.append(scores, ignore_index=True)
    return result


# ### Model Training

# In[110]:


X = patients.drop(["stroke"], axis=1)
y = patients.stroke.astype('int')


# In[111]:


k_folds = 10
cv_results = get_accuracys([None], X, y, k_folds)


# In[112]:


result = build_result(cv_results)
result.head()


# In[ ]:


result.to_excel("PredictResult(balanced-adasyn-revised).xlsx", sheet_name="balanced", float_format="%.4f", index=False)


# ### Train Model with Feature Selected

# In[121]:


patients = pd.read_csv('healthcare-dataset-stroke-data-preprocessed.csv')
feature_selected = patients.copy()
# feature_selected = patients[["age", "avg_glucose_level", "bmi", "stroke"]]
X = feature_selected.drop(["stroke"], axis=1)
y = feature_selected.stroke
feature_selected.sample(5)


# In[122]:


selected_feature_cv_results = get_accuracys(['GMM'], X, y, k_folds)
result = build_result(selected_feature_cv_results)


# In[123]:


result.head()


# In[124]:


result.to_excel("PredictResult(GMM).xlsx", float_format="%.4f", index=False)


# In[98]:


patients = pd.read_csv('data-preprocessed(standardization).csv')
# feature_selected = patients[["age", "hypertension", "avg_glucose_level", "bmi", "stroke"]]
feature_selected = patients[["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi", "smoking_status", "work_type_Self-employed", "stroke"]]
X = feature_selected.drop(["stroke"], axis=1)
y = feature_selected.stroke
feature_selected.sample(5)


# In[99]:


feature_selected.stroke.value_counts()


# In[100]:


selected_feature_cv_results = get_accuracys(['GMM'], X, y, k_folds)
result = build_result(selected_feature_cv_results)


# In[101]:


result.head()


# In[ ]:


result.to_excel("PredictResult().xlsx", float_format="%.4f", index=False)


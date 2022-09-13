#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


# ### Read Source Data

# In[4]:


patients = pd.read_csv('healthcare-dataset-stroke-data.csv')
patients = patients.drop('id', axis=1)
patients.head(10)


# ### Data Labeling

# In[5]:


def label_data(data):
    df = data.copy()
    def encode_bool(x):
        return True if x==1 else False
    df.hypertension = df.hypertension.apply(encode_bool)
    df.heart_disease = df.heart_disease.apply(encode_bool)
    df.stroke = df.stroke.apply(encode_bool)
    
    df.ever_married = df.ever_married == "Yes"
    df['is_male'] = df.gender == "Male"
    df['is_urban'] = df['Residence_type'] == "Urban"
    df.drop(columns = ['gender', 'Residence_type'], inplace=True)
    df.smoking_status = df.smoking_status.map({"Unknown":np.NaN, "never smoked":0, "formerly smoked":1, "smokes":2})
    return df

labeled_data = label_data(patients)
labeled_data.head(10)


# In[6]:


onehot_encoded = pd.get_dummies(labeled_data, columns=["work_type"])
onehot_encoded.head(10)


# ### Data Transfomation & Dealing with Missing Values by KNN

# In[7]:


onehot_encoded.isnull().sum()


# In[8]:


min_max_scaler = MinMaxScaler()
normalized = min_max_scaler.fit_transform(onehot_encoded)


# In[9]:


imputer = KNNImputer(n_neighbors=3)
filled = imputer.fit_transform(normalized)


# In[10]:


preprocessed = pd.DataFrame(filled, columns=onehot_encoded.columns)
preprocessed.head(5)


# In[11]:


preprocessed.isnull().sum()


# In[22]:


preprocessed.stroke.value_counts()


# In[12]:


preprocessed.to_csv('healthcare-dataset-stroke-data-preprocessed.csv', index=False)


# In[20]:


plt.figure(figsize=[20, 20])
hm = sb.heatmap(preprocessed.corr(), annot=True)


# In[16]:


figure = hm.get_figure()    
figure.savefig('conf.png', dpi=300)


# In[23]:


visualization = sb.pairplot(preprocessed, hue="stroke")


# In[25]:


visualization.savefig("pairplot.png", dpi=300)


# In[27]:


visualization = sb.pairplot(labeled_data, hue="stroke")


# In[28]:


visualization.savefig("pairplot_label.png", dpi=300)


# In[43]:


feature_seleted = preprocessed[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status", "stroke"]]
feature_seleted.sample(10, random_state=214)


# In[44]:


visualization_feature_seleted = sb.pairplot(feature_seleted, hue="stroke")


# In[40]:


visualization_feature_seleted.savefig("pairplot_feature_selected.png", dpi=300)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[3]:


from imblearn.over_sampling import ADASYN


# ### Read Source Data

# In[4]:


patients = pd.read_csv('healthcare-dataset-stroke-data-preprocessed.csv')
patients.head(10)


# ### Data Unbalance

# In[5]:


patients.stroke.value_counts()


# ### Data Augmentation (ADASYN)

# In[16]:


X = patients.drop(["stroke"], axis=1)
y = patients.stroke.astype('int')
adasyn = ADASYN(random_state=214, n_neighbors=3)
X_balance, y_balance = adasyn.fit_resample(X, y)


# In[18]:


from collections import Counter
Counter(y_balance)


# In[20]:


balanced_patients = pd.DataFrame(X_balance, columns=patients.columns)
balanced_patients["stroke"] = y_balance
print(balanced_patients.shape)
balanced_patients.head()


# In[21]:


data = [stroke_patients.mean(), stroke_balanced_patients.mean(), stroke_patients.var(), stroke_balanced_patients.var()]
diff = pd.DataFrame(data, columns=patients.columns)
diff.insert(loc=0, column='Note', value=["original_mean", "augmentation_mean", "original_variance", "augmentation_variance"])
diff


# In[11]:


diff.to_csv('diff_augmentation_balance(ADASYN).csv', float_format='%.4f', index=False)


# In[17]:


balanced_patients.to_csv('data-preprocessed-augmentation(ADASYN).csv', index=False)


# In[12]:


plt.figure(figsize=[20, 20])
hm = sb.heatmap(balanced_patients.corr(), annot=True)


# In[13]:


figure = hm.get_figure()    
figure.savefig('conf-balanced.png', dpi=300)


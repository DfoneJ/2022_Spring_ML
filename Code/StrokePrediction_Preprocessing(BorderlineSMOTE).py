#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[72]:


from imblearn.over_sampling import BorderlineSMOTE


# ### Read Source Data

# In[73]:


patients = pd.read_csv('healthcare-dataset-stroke-data-preprocessed.csv')
patients.head(10)


# ### Data Augmentation (BorderlineSMOTE)

# In[92]:


smote = BorderlineSMOTE(random_state=214, k_neighbors=5)
X_balance, y_balance = adasyn.fit_resample(X, y)


# In[93]:


from collections import Counter
Counter(y_balance)


# In[87]:


balanced_patients = pd.DataFrame(X_balance, columns=patients.columns)
balanced_patients["stroke"] = y_balance
print(balanced_patients.shape)
balanced_patients.head()


# In[88]:


stroke_patients = patients[patients.stroke == 1]
stroke_balanced_patients = balanced_patients[balanced_patients.stroke == 1]
print(len(stroke_patients), len(stroke_balanced_patients))


# In[89]:


data = [stroke_patients.mean(), stroke_balanced_patients.mean(), stroke_patients.var(), stroke_balanced_patients.var()]
diff = pd.DataFrame(data, columns=patients.columns)
diff.insert(loc=0, column='Note', value=["original_mean", "augmentation_mean", "original_variance", "augmentation_variance"])
diff


# In[81]:


diff.to_csv('diff_augmentation_balance(BorderlineSMOTE).csv', float_format='%.4f', index=False)


# In[82]:


balanced_patients.to_csv('data-preprocessed-augmentation(BorderlineSMOTE).csv', index=False)


# In[83]:


plt.figure(figsize=[20, 20])
hm = sb.heatmap(balanced_patients.corr(), annot=True)


# In[84]:


figure = hm.get_figure()    
figure.savefig('conf-balanced(BorderlineSMOTE).png', dpi=300)


# In[ ]:





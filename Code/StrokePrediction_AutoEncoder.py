#!/usr/bin/env python
# coding: utf-8

# ### Import

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, roc_curve, auc, precision_recall_curve
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint


# ### Function

# In[2]:


def label_encoding(data):
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
    df.smoking_status = df.smoking_status.map({"Unknown":0, "never smoked":1, "formerly smoked":2, "smokes":3})
    return df


# In[3]:


def get_MSE_MAE(X_val, prediction_non_stroke, stroke_dataset, prediction_stroke):
    mse_val = np.mean(np.power(X_val  - prediction_non_stroke, 2), axis=1)
    mse_predict = np.mean(np.power(stroke_dataset - prediction_stroke, 2), axis=1)
    mae_val = np.mean(np.abs(X_val - prediction_non_stroke), axis=1)
    mae_predict = np.mean(np.abs(stroke_dataset - prediction_stroke), axis=1)

    df_MSE_MAE = pd.DataFrame()
    df_MSE_MAE['stroke'] = [0]*len(mse_val) + [1]*len(mse_predict)
    df_MSE_MAE['MSE'] = np.hstack([mse_val, mse_predict])
    df_MSE_MAE['MAE'] = np.hstack([mae_val, mae_predict])
    df_MSE_MAE = df_MSE_MAE.sample(frac=1).reset_index(drop=True)
    return df_MSE_MAE


# In[4]:


def AE_draw_MSE_MAE(label1, label2, class_name, dataframe_mse_mae):
    markers = ['o', '^']
    colors = ['dodgerblue', 'coral']
    labels = [label1, label2]
    plt.figure(figsize=(14, 5))

    plt.subplot(121) # 左畫板
    for flag in [1, 0]:
        temp = dataframe_mse_mae[dataframe_mse_mae[class_name] == flag]
        plt.scatter(temp.index, temp['MAE'], alpha=0.7, marker=markers[flag], c=colors[flag], label=labels[flag])
    plt.title('Reconstruction MAE')
    plt.ylabel('Reconstruction MAE')
    plt.xlabel('Index')

    plt.subplot(122) # 又畫板
    for flag in [1, 0]:
        temp = dataframe_mse_mae[dataframe_mse_mae[class_name] == flag]
        plt.scatter(temp.index, temp['MSE'], alpha=0.7, marker=markers[flag], c=colors[flag], label=labels[flag])
    plt.legend(loc=[1, 0], fontsize=12)
    plt.title('Reconstruction MSE')
    plt.ylabel('Reconstruction MSE')
    plt.xlabel('Index')
    
    plt.show()


# In[5]:


def get_metric_scores(predict, actual):
    scores = {}
    scores['accuracy'] = accuracy_score(actual, predict)
    scores['precision'] = precision_score(actual, predict)
    scores['recall'] = recall_score(actual, predict)
    scores['f1'] = f1_score(actual, predict)
    scores['ROC'] = roc_auc_score(actual, predict)
    return scores


# In[34]:


def draw_AE_ROCs(MAE_MSE_1, MAE_MSE_2, MAE_MSE_3):
    plt.figure(figsize=(14, 22))
    
    # Model 1
    plt.subplot(321)
    fpr, tpr, _ = roc_curve(MAE_MSE_1['stroke'], MAE_MSE_1['MAE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MAE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')
    
    plt.subplot(322)
    fpr, tpr, _ = roc_curve(MAE_MSE_1['stroke'], MAE_MSE_1['MSE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MSE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')

    #Model 2
    plt.subplot(323)
    fpr, tpr, _ = roc_curve(MAE_MSE_2['stroke'], MAE_MSE_2['MAE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MAE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')
    
    plt.subplot(324)
    fpr, tpr, _ = roc_curve(MAE_MSE_2['stroke'], MAE_MSE_2['MSE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MSE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')

    #Model 3
    plt.subplot(325)
    fpr, tpr, _ = roc_curve(MAE_MSE_2['stroke'], MAE_MSE_2['MAE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MAE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')
    
    plt.subplot(326)
    fpr, tpr, _ = roc_curve(MAE_MSE_2['stroke'], MAE_MSE_2['MSE'])
    roc_auc = auc(fpr, tpr)
    plt.title('ROC for AutoEncoder1 based on MSE\nAUC = %0.2f'%(roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('True Positive'); plt.xlabel('False Positive')
    
    plt.show()


# In[7]:


def biuld_AutoEncoder_with_cross_validation(AutoEncoder, dataset):
    print("dataset :", dataset.shape)
    stroke_dataset = dataset[(dataset['stroke'] == 1.0)]
    stroke_dataset = stroke_dataset.drop(["stroke"], axis=1)
    print("stroke dataset :", stroke_dataset.shape)
    non_stroke_dataset = dataset[(dataset['stroke'] == 0.0)]
    y = non_stroke_dataset['stroke']
    non_stroke_dataset = non_stroke_dataset.drop(["stroke"], axis=1)
    print("non stroke dataset :", non_stroke_dataset.shape)
    print("y for non stroke dataset :", y.shape)
    print("======================================")
    metrics = ['accuracy', 'precision', 'recall','f1', 'ROC']
    MAE_cv_results = {k:np.array([]) for k in metrics}
    MSE_cv_results = {k:np.array([]) for k in metrics}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf_splited = skf.split(non_stroke_dataset, y)
    
    for fit_index, predict_index in skf_splited:
        
        X_fit, X_predict = non_stroke_dataset.iloc[fit_index], non_stroke_dataset.iloc[predict_index]
        X_fit_train, X_fit_test = train_test_split(X_fit, test_size=0.25)
        print("X_fit_train for CV:", X_fit_train.shape)
        print("X_fit_test for CV:", X_fit_test.shape)
        AutoEncoder.fit(X_fit_train, X_fit_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_fit_test ,X_fit_test), verbose=0)
        
        print("X_predict for predict:", X_predict.shape)
        non_stroke_predict = AutoEncoder.predict(X_predict)
        print("stroke_dataset for predict:", stroke_dataset.shape)
        stroke_predict = AutoEncoder.predict(stroke_dataset)
        df_MSE_MAE = get_MSE_MAE(X_predict, non_stroke_predict, stroke_dataset, stroke_predict)
        AE_draw_MSE_MAE('Non-Stroke', 'Stroke', 'stroke', df_MSE_MAE)
        Stroke_MSE_MAE = df_MSE_MAE[(df_MSE_MAE['stroke'] == 1.0)]
        NonStroke_MSE_MAE = df_MSE_MAE[(df_MSE_MAE['stroke'] == 0.0)]
        threshold = (Stroke_MSE_MAE.mean() + NonStroke_MSE_MAE.mean())/2
        df_MSE_MAE['MAE_predict'] = np.where(df_MSE_MAE['MAE']>threshold['MAE'], 1, 0)
        df_MSE_MAE['MSE_predict'] = np.where(df_MSE_MAE['MSE']>threshold['MSE'], 1, 0)
        
        MAE_scores = get_metric_scores(df_MSE_MAE['MAE_predict'], df_MSE_MAE['stroke'])
        MSE_scores = get_metric_scores(df_MSE_MAE['MSE_predict'], df_MSE_MAE['stroke'])
        
        for score in MAE_scores.keys():
            MAE_cv_results[score] = np.append(MAE_cv_results[score], MAE_scores[score])
            MSE_cv_results[score] = np.append(MSE_cv_results[score], MSE_scores[score])
            
    MAE_result = pd.DataFrame([], columns=[*metrics])
    MSE_result = pd.DataFrame([], columns=[*metrics])
    for score in metrics:
        MAE_result[score] = MAE_cv_results[score]
        MSE_result[score] = MSE_cv_results[score]
    print("\nMAE_scores :")
    print(MAE_result.mean())
    print("\nMSE_scores :")
    print(MSE_result.mean())
    return df_MSE_MAE


# In[8]:


def AE_draw_Precision_Recall_curve(class_name, dataframe_mse_mae):
    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(['MAE', 'MSE']):
        plt.subplot(1, 2, i+1)
        precision, recall, _ = precision_recall_curve(dataframe_mse_mae[class_name], dataframe_mse_mae[metric])
        pr_auc = auc(recall, precision)
        plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
        plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
        plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.show()


# In[9]:


def AE_draw_ROC(class_name, dataframe_mse_mae):
    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(['MAE', 'MSE']):
        plt.subplot(1, 2, i+1)
        fpr, tpr, _ = roc_curve(dataframe_mse_mae[class_name], dataframe_mse_mae[metric])
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f'%(metric, roc_auc))
        plt.plot(fpr, tpr, c='coral', lw=4)
        plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
        plt.ylabel('TPR'); plt.xlabel('FPR')
    plt.show()


# ### Dataset

# In[10]:


original_patients = pd.read_csv('healthcare-dataset-stroke-data.csv')
original_patients = original_patients.drop(["id"], axis=1)

other_gender = original_patients[(original_patients['gender'] == 'Other')]
unknown_smoking_status = original_patients[(original_patients['smoking_status'] == 'Unknown')]

print(original_patients.isna().sum(),'\n')
print('other gender\t\t', other_gender.shape[0])
print('unknown smoking status\t', unknown_smoking_status.shape[0])


# In[11]:


original_patients = original_patients[(original_patients['gender'] != 'Other')]
original_patients.shape


# #### Model 1. Drop Missing Data + Normalization =====================================================================

# In[12]:


model_1_dataset = original_patients.dropna() # drop na bmi rows
model_1_dataset = label_encoding(model_1_dataset) # label encoding
model_1_dataset = pd.get_dummies(model_1_dataset, columns=["work_type"]) # one-hot encoding
print(model_1_dataset.shape)
model_1_dataset.sample(3)


# In[13]:


# Normalize
min_max_scaler = MinMaxScaler()
model_1_normalize = min_max_scaler.fit_transform(model_1_dataset)
model_1_dataset = pd.DataFrame(model_1_normalize, columns=model_1_dataset.columns)
model_1_dataset.sample(3)


# #### Model 2. Normalization + KNN Imputer =========================================================================

# In[14]:


model_2_dataset = label_encoding(original_patients) # label encoding
model_2_dataset = pd.get_dummies(model_2_dataset, columns=["work_type"]) # one-hot encoding
print(model_2_dataset.shape)
model_2_dataset.sample(3)


# In[15]:


# Normalize + Imputation
min_max_scaler = MinMaxScaler()
model_2_normalize = min_max_scaler.fit_transform(model_2_dataset)
imputer = KNNImputer(n_neighbors=3)
model_2_impute = imputer.fit_transform(model_2_normalize)
model_2_dataset = pd.DataFrame(model_2_impute, columns=model_2_dataset.columns)
model_2_dataset.sample(3)


# #### Model 3. Drop Missing Data + Normalization + Feature Selection ====================================================

# In[16]:


model_3_dataset = model_1_dataset[["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi", "smoking_status", "work_type_Self-employed", "stroke"]]
print(model_3_dataset.shape)
model_3_dataset.sample(3)


# ### Build Model

# #### Model 1

# In[17]:


# parameters
input_dim = model_1_dataset.shape[1]-1
encoding_dim = 8
# model
input_layer1 = Input(shape=(input_dim,))
encoder1 = Dense(encoding_dim, activation="tanh")(input_layer1)
encoder1 = Dense(int(encoding_dim / 2), activation="relu")(encoder1)
decoder1 = Dense(int(encoding_dim / 2), activation='tanh')(encoder1)
decoder1 = Dense(input_dim, activation='relu')(decoder1)
autoencoder1 = Model(inputs=input_layer1, outputs=decoder1)
# compile model
autoencoder1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[18]:


autoencoder1.summary()


# In[19]:


MSE_MAE_1 = biuld_AutoEncoder_with_cross_validation(autoencoder1, model_1_dataset)


# #### Model 2

# In[20]:


# parameters
input_dim = model_2_dataset.shape[1]-1
encoding_dim = 8
# model
input_layer2 = Input(shape=(input_dim,))
encoder2 = Dense(encoding_dim, activation="tanh")(input_layer2)
encoder2 = Dense(int(encoding_dim / 2), activation="relu")(encoder2)
decoder2 = Dense(int(encoding_dim / 2), activation='tanh')(encoder2)
decoder2 = Dense(input_dim, activation='relu')(decoder2)
autoencoder2 = Model(inputs=input_layer2, outputs=decoder2)
# compile model
autoencoder2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[21]:


autoencoder2.summary()


# In[22]:


MSE_MAE_2 = biuld_AutoEncoder_with_cross_validation(autoencoder2, model_2_dataset)


# #### Model 3

# In[23]:


# parameters
input_dim = model_3_dataset.shape[1]-1
encoding_dim = 5
# model
input_layer3 = Input(shape=(input_dim,))
encoder3 = Dense(encoding_dim, activation="tanh")(input_layer3)
encoder3 = Dense(int(3), activation="relu")(encoder3)
decoder3 = Dense(int(3), activation='tanh')(encoder3)
decoder3 = Dense(input_dim, activation='relu')(decoder3)
autoencoder3 = Model(inputs=input_layer3, outputs=decoder3)
# compile model
autoencoder3.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[24]:


autoencoder3.summary()


# In[25]:


MSE_MAE_3 = biuld_AutoEncoder_with_cross_validation(autoencoder3, model_3_dataset)


# ### Compare Three Models' ROC

# In[35]:


draw_AE_ROCs(MSE_MAE_1, MSE_MAE_2, MSE_MAE_3)


# ### Others

# In[27]:


# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.plot(history['loss'], c='dodgerblue', lw=3)
# plt.plot(history['val_loss'], c='coral', lw=3)
# plt.title('model loss')
# plt.ylabel('mse'); plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')

# plt.subplot(122)
# plt.plot(history['mae'], c='dodgerblue', lw=3)
# plt.plot(history['val_mae'], c='coral', lw=3)
# plt.title('model mae')
# plt.ylabel('mae'); plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right');


# In[28]:


# AE_draw_Precision_Recall_curve('stroke', df_MSE_MAE)


# In[29]:


# AE_draw_ROC('stroke', df_MSE_MAE)


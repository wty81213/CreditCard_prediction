#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Preprocessing import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import math

from sklearn.externals import joblib


## matplotlib,seaborn,pyecharts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.style.use('ggplot') #風格設置近似R這種的ggplot庫
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno


# In[2]:


path = 'D:/Emily/Competition/T-brain/CreditCard_FraudDetection/'


# ## 匯入原始訓練、測試資料

# In[3]:


df_train = pd.read_csv(path + 'dataset/original/train.csv')
df_test = pd.read_csv(path + 'dataset/original/test.csv')


# In[4]:


txkey = df_test['txkey']


# In[5]:


print('df_train.shape' + str(df_train.shape))
df_train.head()


# In[6]:


print('df_test.shape' + str(df_test.shape))
df_test.head()


# ## Data Preprocessing-1

# In[5]:


# loctm的時間轉換
df_train = h_m_s(data= df_train ,col='loctm')
df_test = h_m_s(data= df_test ,col='loctm')

#欄位轉類別
df_train = col_to_cat(data= df_train)
df_test = col_to_cat(data= df_test)


# ## 對訓練、驗證資料做EDA

# In[8]:


#output敘述性統計
# df_train.describe(include = 'all').T.to_excel(path + EDA/df_train_describe.xlsx')
# df_test.describe(include = 'all').T.to_excel(path + EDA/df_test_describe.xlsx')


# # 訓練與測試的交集交易卡號約41%
# df_m = pd.merge(df_train[['cano']], df_test[['cano','conam']], on = ['cano'], how = 'left')
# df_m[df_m['conam'] >=0] #634,410 rows #100*634410/1521787= 41.68%


# # missing
# missing_pct(df_train)

# 	Total	Percentage(%)
# flbmk	12581	0.826725
# flg_3dsmk	12581	0.826725


# missing_pct(df_test)
# 	Total	Percentage(%)
# flbmk	3715	0.881031
# flg_3dsmk	3715	0.881031


# ### Describe Target

# In[9]:


# 查看目標列的情況
print(df_train.groupby('fraud_ind').size())
print('fraud_ind=0: ' + str(len(df_train[df_train['fraud_ind']==0]) /len(df_train)))

# 目標變量分布可視化
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='fraud_ind',data=df_train,ax=axs[0])
axs[0].set_title("Frequency of each Class")
df_train['fraud_ind'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Class")
plt.show()


# ## Data Preprocessing-2

# In[6]:


#刪除無意義欄位
df_train = df_train.drop(columns = ['locdt','loctm','txkey','loctm_S', 'loctm_M'])
df_test = df_test.drop(columns = ['locdt','loctm','txkey','loctm_S', 'loctm_M'])

#處理missing data
df_train = df_train.fillna('Missing')
df_test = df_test.fillna('Missing')

print('missing_pct(df_train):\n' + str(missing_pct(df_train)) + '\n')
print('missing_pct(df_test):\n' + str(missing_pct(df_test)) + '\n')

#欄位轉類別
df_train = col_to_cat(data= df_train)
df_test = col_to_cat(data= df_test)

print('df_train.info():')
print(df_train.info())
print('df_test.info():')
print(df_test.info())


# In[ ]:





# ## 類別欄位做WOE

# In[ ]:


# calculate woe using train dataframe
compute_woe_iv(data = df_train ,var_list = var_list ,Y_flag = Y_flag)


# In[8]:


# df_train文字型欄位各類別值,產出對應的WOE、WOE平均值
df_woe_mean = woe_mean()

# df_train的文字型欄位值轉為WOE值
df_train_woe = df_WOE(data = df_train)

# df_test的文字型欄位值轉為WOE值
df_test_woe = df_WOE(data = df_test)

df_woe_mean.head()


# In[9]:


#找出df_test有遺漏值欄位有哪些,並以該欄位在df_woe_mean的平均值填補遺漏值
print('missing_pct(data = df_train_woe):\n' + str(missing_pct(data = df_train_woe)))
print('missing_pct(data = df_test_woe):\n' + str(missing_pct(data = df_test_woe)))
missing_pct_test = missing_pct(data = df_test_woe)

test_missing_list= list(missing_pct_test[missing_pct_test['Total'] > 0].index)
df_test_missing = df_woe_mean[df_woe_mean['Variable'].isin(test_missing_list)][['Variable','woe_mean']].drop_duplicates()
print(test_missing_list)
print(df_test_missing)


# In[10]:


df_test_woe_fillna = pd.DataFrame()
for i in test_missing_list:
    val = df_test_missing[df_test_missing['Variable'] == i]['woe_mean'].values[0]
    val_fillna = df_test_woe[i].fillna(val)
    df_test_woe_fillna[i] = val_fillna

df_test_woe_fillna


# In[11]:


df_test_woe = df_test_woe.drop(columns = test_missing_list)
df_test_woe = pd.concat([df_test_woe, df_test_woe_fillna], axis = 1)

print(df_train_woe.columns)
print(df_test_woe.columns)

df_test_woe.head()


# In[ ]:





# ## 數值欄位做標準化 - 'conam', 'iterm'

# In[12]:


# df_train 數值資料做MinMaxScaler
df_train_num = df_train[['conam','iterm']]
df_train_std = numerical_columns_handle(dt = df_train_num, save_scaler=True)
df_train_std = pd.DataFrame(data = df_train_std, columns = ['conam','iterm'] )

# df_train 轉WOE的欄位 與 轉MinMaxScaler 的欄位做合併
X =pd.concat([df_train_woe, df_train_std], axis = 1)
y = df_train[['fraud_ind']]
# pd.concat([y , X ], axis = 1).to_csv(path + 'dataset/transformed/df_train_trans.csv')




# df_test 數值資料做MinMaxScaler
scaler = preprocessing.MinMaxScaler()
dt = df_train[['conam','iterm']]
scaler.fit_transform(dt)
# joblib.dump(scaler, path + 'models/numerical_minmaxscaler.pkl')


df_test_num = df_test[['conam','iterm']]
df_test_std = scaler.transform(df_test_num)

conam = []
iterm = []
for i in range(0,len(df_test_std)):
    a = df_test_std[i][0]
    b = df_test_std[i][1]
    conam.append(a)
    iterm.append(b)


df_test_std = pd.DataFrame()
df_test_std['conam'] = conam
df_test_std['iterm'] = iterm

# df_test 轉WOE的欄位 與 轉MinMaxScaler 的欄位做合併
df_test_trans =pd.concat([df_test_woe, df_test_std], axis = 1)
# df_test_trans.to_csv(path + 'dataset/transformed/df_test_trans.csv')


# In[ ]:





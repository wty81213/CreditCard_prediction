import numpy as np
import pandas as pd
import math

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


path = 'D:/Emily/Competition/T-brain/CreditCard_FraudDetection/'

## 訓練、測試資料的loctm 轉為6碼的時分秒
def h_m_s(data, col):
    h_list = []
    m_list = []
    s_list = []

    for i in data[col]:
        i = str(int(i))
        val_time = str(0)*(6-len(i)) + i

        val_h = val_time[0:2]
        val_m = val_time[2:4]
        val_s = val_time[4:6]

        h_list.append(val_h)
        m_list.append(val_m)
        s_list.append(val_s)

    data['loctm_H'] = h_list
    data['loctm_M'] = m_list
    data['loctm_S'] = s_list

    return data


## 欄位轉類別變數
def col_to_cat(data) :
    if 'fraud_ind' not in data.columns:
        data_chtype = data[['acqic','bacno','cano','contp','csmcu','etymd','hcefg','mcc','mchno','scity','stocn','stscd','ecfg','flbmk','flg_3dsmk','insfg','ovrlt']].astype('object')  #,'txkey'
        data = data.drop(columns = ['acqic','bacno','cano','contp','csmcu','etymd','hcefg','mcc','mchno','scity','stocn','stscd','ecfg','flbmk','flg_3dsmk','insfg','ovrlt']) #,'txkey'

    else:
        data_chtype = data[['acqic','bacno','cano','contp','csmcu','etymd','fraud_ind','hcefg','mcc','mchno','scity','stocn','stscd','ecfg','flbmk','flg_3dsmk','insfg','ovrlt']].astype('object') #,'txkey'
        data = data.drop(columns = ['acqic','bacno','cano','contp','csmcu','etymd','fraud_ind','hcefg','mcc','mchno','scity','stocn','stscd','ecfg','flbmk','flg_3dsmk','insfg','ovrlt']) #,'txkey'


    data = pd.concat([data,data_chtype], axis = 1)

    return data


## 處理Missing Data
#missing data percentage
def missing_pct(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (100*data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage(%)'])
    return missing_data



## 數值欄位做標準化
def numerical_columns_handle(dt, save_scaler=True):
    scaler = MinMaxScaler()

    dt = scaler.fit_transform(dt)

    if save_scaler:
        joblib.dump(scaler, path + 'models/numerical_minmaxscaler.pkl')

    return dt



## 類別欄位做WOE
var_list = [ 'ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt',
       'loctm_H', 'acqic', 'bacno', 'cano', 'contp',
       'csmcu', 'etymd', 'hcefg', 'mcc', 'mchno', 'scity',
       'stocn', 'stscd']
Y_flag = 'fraud_ind'


# 無合併群組,可產出各類變數的WOE值
# 對訓練資料做處理
def compute_woe_iv(data,var_list,Y_flag):
    df_type = 'train'
    df = data
    var_num = len(var_list)
    totalG_B = df.groupby([Y_flag])[Y_flag].count()  # 计算正负样本多少个
    B = totalG_B[1]
    G = totalG_B[0]
    woe_all = np.zeros((1, 8))
    var_iv = np.zeros((var_num))
    data_index = []
    for k in range(0, var_num):
        var1 = df.groupby([var_list[k]])[Y_flag].count()  # 计算col每个分组中的组的个数
        var_class = var1.shape[0]
        woe = np.zeros((var_class, 8))
        woe_pre = pd.DataFrame(data={'x1': [], 'ifbad': [],'values' : []})
        total = df.groupby([var_list[k], Y_flag])[Y_flag].count()  # 计算该变量下每个分组响应个数
        total1 = pd.DataFrame({'total': total})
        mu = []
        for u,group in df.groupby([var_list[k], Y_flag])[Y_flag]:
            mu.append(list(u))
        print(mu)
        for lab1 in total.index.levels[0]:
            for lab2 in total.index.levels[1]:
                print(lab1,lab2)
    #             temporary = pd.DataFrame(data={'x1': [lab1], 'ifbad': [lab2], 'values': [1]})
                if [lab1,lab2] not in mu:
                    temporary = pd.DataFrame(data={'x1': [lab1], 'ifbad': [lab2], 'values' : [0]})
                else:
                    temporary = pd.DataFrame(data={'x1': [lab1], 'ifbad': [lab2], 'values' : [total1.xs((lab1, lab2)).values[0]]})
                woe_pre = pd.concat([woe_pre, temporary])
            #print(woe_pre)
        woe_pre.set_index(['x1','ifbad'], inplace=True)
        print(woe_pre)

        # 计算 WOE
        for i in range(0, var_class):   #var_class
            woe[i,0] = woe_pre.values[2 * i + 1]
            woe[i,1] = woe_pre.values[2 * i]
            woe[i,2] = woe[i,0] + woe[i,1]
            woe[i, 3] = woe_pre.values[2 * i + 1] / B  # pyi
            woe[i, 4] = woe_pre.values[2 * i] / G  # pni
            abb = lambda i:(math.log(woe[i, 3] / woe[i, 4])) if woe[i, 3] != 0 else 0 # 防止 ln 函数值域报错
            woe[i, 5] = abb(i)
            woe[np.isinf(woe)] = 0  #将无穷大替换为0，参与计算 woe 计算


            woe[i, 6] = (woe[i, 3] - woe[i, 4]) * woe[i, 5]  # iv_part
            var_iv[k] += woe[i, 6]
        iv_signal = np.zeros((1,8))
        iv_signal[0,7] =var_iv[k]
        woe_all = np.r_[woe_all, woe,iv_signal]
        index_var = df.groupby([var_list[k]])[Y_flag].count()
        u = index_var.index.values.tolist()
        data_index += u
        data_index += [var_list[k]]
    woe_all = np.delete(woe_all,0,axis=0)
    result = pd.DataFrame(data = woe_all,columns=['bad', 'good', 'class_sum','pyi','pni','woe','iv_part','iv'])
    result.index = data_index

    result.to_csv(path + 'WOE/' + df_type + '/WOE_IV.csv')
    return {print(result)}


WOE_IV = pd.read_csv(path + 'WOE/' + 'train' + '/WOE_IV.csv')
WOE_IV['Value'] = WOE_IV['Unnamed: 0'].astype('str')
WOE_IV['Key'] = WOE_IV['Unnamed: 0']
WOE_IV = WOE_IV.drop(columns = ['Unnamed: 0'])
df_var = pd.DataFrame(columns = ['Variable'])
df_var['Variable'] = var_list
df_var['Key'] = var_list
WOE_IV = pd.merge(WOE_IV, df_var, on = ['Key','Key'], how = 'left' )
WOE_IV['Variable_1'] = WOE_IV['Variable'].fillna(method = 'bfill')
WOE_IV = WOE_IV[['Variable_1','Value','woe']]

# df_train文字型欄位各類別值,產出對應的WOE、WOE平均值
def woe_mean():
    df_woe_mean = pd.DataFrame()
    for i in var_list:

        c = WOE_IV[WOE_IV['Variable_1'] == i][['Variable_1','Value','woe']]
        c['Variable'] = 'WOE_' + c['Variable_1']
        c = c[~ c['Value'].isin(var_list)]
        c['woe_mean'] = sum(c['woe'])/(len(c['Value']))
        c = c[['Variable','Value','woe','woe_mean']]
        df_woe_mean = df_woe_mean.append(c)

    df_woe_mean.to_csv(path + 'WOE/' + 'train' + '/df_woe_mean.csv')
    return df_woe_mean


# data set的文字型欄位值轉為WOE值
def df_WOE(data):
    df_woe = pd.DataFrame()
    for i in var_list:

        a = data[[i]].astype('str')
        a['Value'] = a[i]
        b = WOE_IV[WOE_IV['Variable_1'] == i][['Value','woe']]
        WOE_NM = 'WOE_'+ i
        WOE_Value = pd.merge(a, b, on = ['Value', 'Value'], how = 'left')['woe']
        df_woe[WOE_NM] = WOE_Value

    return df_woe
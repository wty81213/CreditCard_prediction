import gc
import numpy as np 
import pandas as pd 
from function.utils import *
from function.config import *
from function.for_data_process import *
from function.for_building_model import *
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go 
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')
#import cufflinks as cf
#cf.set_config_file(offline=True,world_readable=True)

# import the dataset
training_set = pd.read_csv('./data/train.csv')
testing_set = pd.read_csv('./data/test.csv')
data_set = pd.concat([training_set,testing_set],axis = 0)
data_set.index = data_set['txkey']
# DS_table(data_set)

del training_set,testing_set
gc.collect()

# config
category_feat = config_setting()['category_feat']
target_col = config_setting()['target_col']

# convert to string
data = convert_to_string(data_set,category_feat)
Descriptive_stat_table = DS_table(data,reading_local_file = True)
# Descriptive_stat_table = DS_table(data,saving_file = True)

data = loctm_process(data,category_feat)

path = './temp_data/'
for  file_name in ['df1.csv','df2.csv','df3.csv','df4.csv']:
    # file_name = 'df2.csv'
    file_path = path + file_name
    temp_data = pd.read_csv(file_path)
    temp_data['bacno'] = temp_data['bacno'].astype(str)
    if file_name == 'df1.csv':
        data = data.merge(temp_data,on = ['bacno','locdt'],how = 'left')
    else :
        data = data.merge(temp_data,on = ['bacno'],how = 'left')
    

data = drop_col_process(data,category_feat)

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
data['hour_region']= label_encoder.fit_transform(data['hour_region']) 

data = convert_cat_into_onehot(data,category_feat,threshold = 10)

data = generate_datatype(data,target_col,category_feat,split_ratio = 0.7)
for col in category_feat:
    data[col] = data[col].astype(np.integer)

split_df = split_dataset(data,target_col)

datasetloader = split_df;cat_feats = category_feat
trainset = datasetloader['train']
validset = datasetloader['valid']
testset =  datasetloader['test']
cat_feats = cat_feats

lgbtrain = lgb.Dataset(data = trainset['X'].values,label=trainset['y'],
                    feature_name=trainset['X'].columns.tolist(),categorical_feature = cat_feats,
                    free_raw_data = False)


from sklearn.metrics import f1_score

best_param = {'boosting': 'gbdt',
                'objective': 'binary', 
                'task':'train',
                'metric': ['auc'],
                'scale_pos_weight':2,
                'n_jobs': -1,
                'bagging_fraction': 0.43310240726832455,
                'feature_fraction': 0.3548379046859415,
                'lambda_l2': 0.675970059229584,
                'max_depth': 7,
                'num_leaves': 19}

lgbtrain = lgb.Dataset(data = trainset['X'].values,label=trainset['y'],
                    feature_name=trainset['X'].columns.tolist(),categorical_feature = cat_feats,
                    free_raw_data = False)

final_model = lgb.train(best_param,lgbtrain,num_boost_round = 300)
predict_probability = final_model.predict(validset['X'])
predict_valid_y = (predict_probability>=0.5).astype('int')
print(f1_score(validset['y'], predict_valid_y))

all_trainset = {'X':pd.concat([trainset['X'],validset['X']],axis = 0),
                        'y':np.append(trainset['y'],validset['y'])}

lgbtrain = lgb.Dataset(data = all_trainset['X'].values,label=all_trainset['y'],
                    feature_name=all_trainset['X'].columns.tolist(),categorical_feature = cat_feats,
                    free_raw_data = False)

F_model = lgb.train(best_param,lgbtrain,num_boost_round = 300)
predict_test_y = (F_model.predict(testset['X'])>=0.5).astype('int')
submit_table = pd.DataFrame({'txkey':testset['X'].index,'fraud_ind':predict_test_y})
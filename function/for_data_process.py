import re
import numpy as np
import pandas as pd
from function.utils import DS_table
#from function.config import category_feat

def drop_col_process(data,category_feat):
    df = data.copy()
    # drop_catfeat =  ['txkey','bacno','cano']
    drop_catfeat =  ['txkey']
    df.drop(columns = drop_catfeat,inplace = True)
    category_feat.difference_update(set(drop_catfeat))
    return(df)

def loctm_process(data,category_feat):
    df = data.copy()
    df['loctm'] = df['loctm'].\
                        astype(int).\
                        astype(str).\
                        str.rjust(6, '0')

    df['loctm_hour'] = df['loctm'].\
                        str.slice(start=0,stop = 2).\
                        astype(int).\
                        astype(str) 

    df['loctm_min'] = df['loctm'].\
                        str.slice(start=2,stop = 4).\
                        astype(int).\
                        astype(str)             

    df['loctm_second'] = df['loctm'].\
                        str.slice(start=4,stop = 6).\
                        astype(int).\
                        astype(str)    
    
    drop_catfeat = ['loctm']
    df = df.drop(columns = drop_catfeat)  
    
    hour_mapping = dict()
    hour_mapping.update(dict(zip(list(range(0,5)),np.repeat('early_morning',24))))
    hour_mapping.update(dict(zip(list(range(5,8)),np.repeat('morning',24))))
    hour_mapping.update(dict(zip(list(range(8,11)),np.repeat('forenoon',24))))
    hour_mapping.update(dict(zip(list(range(11,13)),np.repeat('noon',24))))
    hour_mapping.update(dict(zip(list(range(13,16)),np.repeat('afternoon',24))))
    hour_mapping.update(dict(zip(list(range(16,19)),np.repeat('evening',24))))
    hour_mapping.update(dict(zip(list(range(19,24)),np.repeat('night',24))))
    df['hour_region'] = df['loctm_hour'].astype(int).map(hour_mapping)

    add_catfeat = ['loctm_hour','hour_region','loctm_min','loctm_second']
    category_feat.difference_update(set(drop_catfeat))
    category_feat.update(add_catfeat)
    return(df)

# data,feats,remaining_percent = data,['mcc'],0.001
def building_rare_class(data,feats,remaining_percent=None,remaining_count=None,recursive = False):
    
    def combining_rare_class(data,feat,rare_name):
        df = data.copy()
        count_table = df[feat].value_counts()
        count_table = count_table.reset_index()
        # num_class = count_table.shape[0]
        count_table[feat + '_percent'] = count_table[feat]/count_table[feat].sum()
        cumsum_values = count_table[feat+'_percent'].\
            sort_values(ascending=True).\
            cumsum()
        count_table[feat + '_cumsum'] = cumsum_values
        if remaining_percent!= None:
            count_table['israre'] = count_table[feat + '_cumsum']<remaining_percent
        elif remaining_count!= None:
            count_table['israre'] = count_table[feat]<remaining_count
        rare_table = count_table[count_table['israre'] == True]
        num_rare = rare_table.shape[0]
        print('combining the %d class into rare'%(num_rare))
        rare_index = rare_table['index'].tolist()
        df.loc[df[feat].isin(rare_index),feat] = rare_name
        return(df)
    
    df = data.copy()
    for feat in feats:
        # feat = feats[0]
        print('handling %s'%(feat))
        if recursive:
            not_unseen_cat = True
            iterative = 1
            while not_unseen_cat:
                print('iterative %d'%(iterative))
                current_rare_name = 'rare_' + str(iterative)
                df = combining_rare_class(df,feat,current_rare_name)
                testing_cond = df['locdt']>90
                cond = ~df[testing_cond][feat].isin(df[~testing_cond][feat])
                iterative += 1
                if np.sum(cond) == 0:
                    not_unseen_cat = False
        else:
            current_rare_name = 'rare'
            df = combining_rare_class(df,feat,current_rare_name)
    return(df)

def convert_cat_into_onehot(data,category_feat,feats = None,threshold = None):
    df = data.copy()
    if threshold != None:
        Descriptive_stat_table = DS_table(df,reading_local_file = True)
        cond = Descriptive_stat_table.class_num <= threshold
        feats = Descriptive_stat_table['name'][cond].tolist()
    else:
        pass

    temp_df = df[feats]
    df = pd.concat([df,pd.get_dummies(temp_df,feats)],axis = 1)
    df.drop(columns = feats,inplace = True)
    category_feat.difference_update(set(feats))
    return(df)



if __name__ == "__main__":
    pass
import numpy as np
from sklearn.model_selection import train_test_split

# split_ratio = 0.7;target_col = 'fraud_ind'
def generate_datatype(data,target_col,category_feat,split_ratio):
    df = data.copy()
    test_cond = df.locdt > 90
    all_train_index = df.index[~test_cond]
    train_index,valid_index = train_test_split(np.array(all_train_index),\
                                train_size  = split_ratio,\
                                stratify = df.loc[~test_cond,target_col],\
                                random_state = 0,\
                                shuffle = True)

    df.loc[test_cond,'datatype'] = 'test'
    df.loc[df.index.isin(train_index),'datatype'] = 'train'
    df.loc[df.index.isin(valid_index),'datatype'] = 'valid'

    print(df['datatype'].value_counts())
    print(df.groupby(['datatype',target_col]).size())
    
    df.drop(columns = 'locdt',inplace = True)
    category_feat.difference_update(set(['locdt']))

    return(df)

def split_dataset(data,target_col):
    df = data.copy()
    train_df = df.query('datatype == "train"').drop(columns = 'datatype')
    valid_df = df.query('datatype == "valid"').drop(columns = 'datatype')
    test_df = df.query('datatype == "test"').drop(columns = 'datatype')
    return {'train':{'X':train_df.drop(columns = target_col),'y':train_df[target_col]},\
            'valid':{'X':valid_df.drop(columns = target_col),'y':valid_df[target_col]},\
            'test':{'X':test_df.drop(columns = target_col)}}

if __name__ == "__main__":
    pass

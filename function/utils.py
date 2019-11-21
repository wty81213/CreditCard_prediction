import numpy as np
import pandas as pd
import re
import os


def filter_cols(data, re_string):
    expression_string = re.compile(re_string)
    col_names = filter(expression_string.search, data.columns.tolist())
    return (list(col_names))


def convert_to_string(data, category_feat):
    df = data.copy()
    # convert_type_mappping_dict = zip(category_feat,\
    #         np.repeat(str,len(category_feat)))
    # df = df.astype(dict(convert_type_mappping_dict))

    df[list(category_feat)] = df[list(category_feat)].astype(str)

    return (df)


def find_unseen_cat(feat, split_cond):
    test_cat = pd.Series(feat[split_cond].unique())
    train_cat = feat[~split_cond].unique()
    unseen_cat_B = ~test_cat.isin(train_cat)
    # unseen_cat = test_cat[unseen_cat_B].to_list()
    unseen_cat = [test_cat[unseen_cat_B]]
    return ({'unseen_cat_N': np.sum(unseen_cat_B), 'unseen_cat': unseen_cat})


# df = data
def DS_table(df, reading_local_file=False, saving_file=False):
    file_name = 'Descriptive_stat_table.csv'
    saving_path = './temp_data/'
    if os.path.isfile(os.path.join(saving_path,
                                   file_name)) & (reading_local_file):
        result = pd.read_csv(os.path.join(saving_path, file_name))
    else:
        is_train = df['locdt'] > 90

        def descriptive_statistics(x):
            result = pd.Series(index = ['na_num','na_ratio',\
                                        'class_num','class_ratio',
                                        'unseen_cat_count','unseen_cat_ratio',\
                                        'min','max','std'])
            # NA
            result['na_num'] = np.sum(x.isnull())
            result['na_ratio'] = result['na_num'] / len(x)
            if x.dtype == np.object:
                result['class_num'] = len(x.unique())
                result['class_ratio'] = len(x.unique()) / len(x)
                unseen_cat_result = find_unseen_cat(x, is_train)
                result['unseen_cat_count'] = unseen_cat_result['unseen_cat_N']
                result['unseen_cat_ratio'] = unseen_cat_result[
                    'unseen_cat_N'] / len(x.unique())
            if x.dtype in [np.int64, np.float64]:
                result['min'] = np.min(x)
                result['max'] = np.max(x)
                result['std'] = np.std(x)
            return (result)

        result = df.apply(descriptive_statistics, axis=0).T
        result = result.reset_index().rename(columns={"index": 'name'})

    if saving_file:
        result.to_csv(os.path.join(saving_path, file_name), index=False)
    return (result)


if __name__ == "__main__":
    pass

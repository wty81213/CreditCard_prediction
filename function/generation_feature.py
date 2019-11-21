import pandas as pd 
from functools import partial

df = data.copy()

key_columns = 'bacno'
timestamp = 'locdt'

C_feat = ['contp','ecfg','etymd','flbmk','flg_3dsmk','hcefg','insfg','ovrlt','stscd','hour_region']
D_feat =  ['acqic','cano','csmcu','mcc','mchno','scity','stocn','txkey']

previous_day = 10

#--------------------

def previous_count(df):

    selected_df = df[[key_columns,timestamp]+D_feat].copy().reset_index(drop = True)
    min_date,max_date = selected_df['locdt'].min(),selected_df['locdt'].max()

    stack_data = pd.DataFrame(columns = selected_df.columns)

    for current_date in range(min_date,(max_date+1)):
        # current_date = 1
        start_date = 1 if (current_date -  previous_day) < 0 else (current_date -  previous_day)
        end_date = (current_date-1)
        condition = (selected_df[timestamp] >= start_date)&(selected_df[timestamp]<=end_date)
        temp_data = selected_df[condition].copy()
        temp_data[timestamp] = current_date
        stack_data = pd.concat([stack_data,temp_data])

        # if current_date == 10 :
        #     break

    previous_count_table = stack_data.groupby([key_columns,timestamp])[D_feat].\
                                nunique()
    previous_count_table.columns = [col+'_day_'+str(previous_day)+'_nunique' for col in previous_count_table.columns]
    previous_count_table = previous_count_table.reset_index()
    return previous_count_table

def count_for_C(df):
    selected_df2 = df[[key_columns]+C_feat].copy()
    selected_df2 = pd.get_dummies(selected_df2,columns = C_feat)
    count_dummy_table  =  selected_df2.groupby('bacno').sum()
    count_dummy_table.columns = [col+'_nunique' for col in count_dummy_table.columns]
    count_dummy_table = count_dummy_table.reset_index()
    return count_dummy_table

def count_for_D(df):
    selected_df3 = df[[key_columns]+D_feat].copy()
    count_table  =  df.groupby('bacno')[D_feat].nunique()
    count_table.columns = [col+'_nunique' for col in count_table.columns]
    count_table = count_table.reset_index()
    return count_table

def numeric_transformation(df):
    def percentil(x,p):
        return x.quantile(p)
    percentil_075 = partial(percentil,p = 0.75)
    percentil_025 = partial(percentil,p = 0.25)
    acount_numeric_feat_table = df.groupby('bacno')['conam'].\
        agg({'mean','median','std','min','max',
            ('percentil_075',percentil_075),('percentil_025',percentil_025)})
    acount_numeric_feat_table.columns = ['conam_'+col for col in acount_numeric_feat_table.columns]
    acount_numeric_feat_table = acount_numeric_feat_table.reset_index()

    return acount_numeric_feat_table

if __name__ == "__main__":
    df = data.copy()
    print('previous_count')
    df1 = previous_count(df)
    df1.to_csv('./temp_data/df1.csv',index = False)
    #df = df.merge(df1,on = key_columns,how = 'left')
    print('count_for_C')
    df2 = count_for_C(df)
    df2.to_csv('./temp_data/df2.csv',index = False)
    #df = df.merge(df2,on = key_columns,how = 'left')
    print('count_for_D')
    gc.collect()
    df3 = count_for_D(df)
    df3.to_csv('./temp_data/df3.csv',index = False)
    #df = df.merge(df3,on = key_columns,how = 'left')
    print('numeric_transformation')
    df4 = numeric_transformation(df)
    df4.to_csv('./temp_data/df4.csv',index = False)
    #df = df.merge(df4,on = key_columns,how = 'left')
    





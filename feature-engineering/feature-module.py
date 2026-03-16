def features_lagdiff(df):
    import pandas as pd
    import gc
    
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    
    df["u_in_sum"] = df.groupby("breath_id")["u_in"].transform("sum")
    df["u_out_sum"] = df.groupby("breath_id")["u_out"].transform("sum")
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df["u_in_cumsum_reverse"] = df["u_in_sum"] - df["u_in_cumsum"]
    
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df["u_in_first"] = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_last"] = df.groupby("breath_id")["u_in"].transform("last")
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
#     df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
#     df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
#     df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
#     df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
#     df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
#     df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
#     df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
#     df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
#     df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
#     df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
#     df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
#     df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
#     df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
#     df['cross2']= df['time_step']*df['u_out']    
    
    df["time_passed"] = df.groupby("breath_id")["time_step"].diff()
    
    g = df.groupby('breath_id')['u_in']
    df['ewm_u_in_mean'] = g.ewm(halflife=10).mean()\
                           .reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = g.ewm(halflife=10).std()\
                          .reset_index(level=0, drop=True)
    df['ewm_u_in_corr'] = g.ewm(halflife=10).corr()\
                           .reset_index(level=0, drop=True)
    
    df['expand_mean'] = g.expanding(2).mean()\
                         .reset_index(level=0, drop=True)
    df['expand_max'] = g.expanding(2).max()\
                        .reset_index(level=0, drop=True)
    df['expand_std'] = g.expanding(2).std()\
                        .reset_index(level=0, drop=True)
    
    df = df.fillna(0)
    
    df.drop(columns=['one','count'], inplace=True)
    
    return df


def features_roll1(df):
    
    import cudf as pd
    import gc

    ###########
    # Roll 5
    ###########
    df["u_in_rolling_mean5"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=5, min_periods=1).mean()["u_in"].reset_index(drop=True)    
    df["u_in_rolling_max5"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=5, min_periods=1).max()["u_in"].reset_index(drop=True)
    df["u_in_rolling_min5"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=5, min_periods=1).min()["u_in"].reset_index(drop=True) 

    df["u_in_rolling_mean5_diff"] = df["u_in"] - df["u_in_rolling_mean5"]
    df["u_in_rolling_max5_diff"] = df["u_in"] - df["u_in_rolling_max5"]
    df["u_in_rolling_min5_diff"] = df["u_in"] - df["u_in_rolling_min5"]
#     df["u_in_rolling_std5_diff"] = df["u_in"] - df["u_in_rolling_std5"] 

    ###########
    # Roll 10
    ###########
    df["u_in_rolling_mean10"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=10, min_periods=1).mean()["u_in"].reset_index(drop=True)    
    df["u_in_rolling_max10"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=10, min_periods=1).max()["u_in"].reset_index(drop=True)
    df["u_in_rolling_min10"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=10, min_periods=1).min()["u_in"].reset_index(drop=True)

    df["u_in_rolling_mean10_diff"] = df["u_in"] - df["u_in_rolling_mean10"]
    df["u_in_rolling_max10_diff"] = df["u_in"] - df["u_in_rolling_max10"]
    df["u_in_rolling_min10_diff"] = df["u_in"] - df["u_in_rolling_min10"]
#     df["u_in_rolling_std10_diff"] = df["u_in"] - df["u_in_rolling_std10"]

    ###########
    # Roll 20
    ###########
    df["u_in_rolling_mean20"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=20, min_periods=1).mean()["u_in"].reset_index(drop=True)    
    df["u_in_rolling_max20"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=20, min_periods=1).max()["u_in"].reset_index(drop=True)
    df["u_in_rolling_min20"] = df[["breath_id", "u_in"]].groupby("breath_id").rolling(window=20, min_periods=1).min()["u_in"].reset_index(drop=True)
    
    df["u_in_rolling_mean20_diff"] = df["u_in"] - df["u_in_rolling_mean20"]
    df["u_in_rolling_max20_diff"] = df["u_in"] - df["u_in_rolling_max20"]
    df["u_in_rolling_min20_diff"] = df["u_in"] - df["u_in_rolling_min20"]
#     df["u_in_rolling_std20_diff"] = df["u_in"] - df["u_in_rolling_std20"]

    return df
    


def features_roll2(df):
    
    import cudf as pd
    import gc

    ###########
    # Roll 5
    ###########
    rollnum=5
    
    temp1 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].mean().iloc[::-1].reset_index(drop=False)
    temp2 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].max().iloc[::-1].reset_index(drop=False)
    temp3 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].min().iloc[::-1].reset_index(drop=False)

    temp1.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_mean_lead{rollnum}'}, inplace=True)
    temp2.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_max_lead{rollnum}'}, inplace=True)
    temp3.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_min_lead{rollnum}'}, inplace=True)
    
    
    temp1['id'] = len(temp1)-temp1['id']
    temp2['id'] = len(temp2)-temp2['id']
    temp3['id'] = len(temp3)-temp3['id']
    
    temp1['breath_id'] = temp1['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp2['breath_id'] = temp2['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp3['breath_id'] = temp3['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)

    temp = pd.merge(temp1, temp2, on=["breath_id","id"], how="left")
    temp = pd.merge(temp, temp3, on=["breath_id","id"], how="left")

    del temp1, temp2, temp3
    _ = gc.collect()


    ###########
    # Roll 10
    ###########
    rollnum=10
    
    temp1 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].mean().iloc[::-1].reset_index(drop=False)
    temp2 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].max().iloc[::-1].reset_index(drop=False)
    temp3 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].min().iloc[::-1].reset_index(drop=False)

    temp1.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_mean_lead{rollnum}'}, inplace=True)
    temp2.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_max_lead{rollnum}'}, inplace=True)
    temp3.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_min_lead{rollnum}'}, inplace=True)
    
    
    temp1['id'] = len(temp1)-temp1['id']
    temp2['id'] = len(temp2)-temp2['id']
    temp3['id'] = len(temp3)-temp3['id']
    
    temp1['breath_id'] = temp1['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp2['breath_id'] = temp2['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp3['breath_id'] = temp3['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    

    temp = pd.merge(temp, temp1, on=["breath_id","id"], how="left")
    temp = pd.merge(temp, temp2, on=["breath_id","id"], how="left")
    temp = pd.merge(temp, temp3, on=["breath_id","id"], how="left")

    del temp1, temp2, temp3
    _ = gc.collect()


    ###########
    # Roll 20
    ###########
    rollnum=20
    
    temp1 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].mean().iloc[::-1].reset_index(drop=False)
    temp2 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].max().iloc[::-1].reset_index(drop=False)
    temp3 = df[["breath_id",'id', "u_in"]].iloc[::-1].reset_index(drop=True).groupby(["breath_id"]).rolling(window=rollnum, min_periods=1)["u_in"].min().iloc[::-1].reset_index(drop=False)

    temp1.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_mean_lead{rollnum}'}, inplace=True)
    temp2.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_max_lead{rollnum}'}, inplace=True)
    temp3.rename(columns={'index':'id',
                        'u_in':f'u_in_rolling_min_lead{rollnum}'}, inplace=True)
    
    
    temp1['id'] = len(temp1)-temp1['id']
    temp2['id'] = len(temp2)-temp2['id']
    temp3['id'] = len(temp3)-temp3['id']
    
    temp1['breath_id'] = temp1['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp2['breath_id'] = temp2['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)
    temp3['breath_id'] = temp3['id'].applymap(lambda x: (x//80)+1 if (x%80)!=0 else x//80)

    
    temp = pd.merge(temp, temp1, on=["breath_id","id"], how="left")
    temp = pd.merge(temp, temp2, on=["breath_id","id"], how="left")
    temp = pd.merge(temp, temp3, on=["breath_id","id"], how="left")

    del temp1, temp2, temp3
    _ = gc.collect()

    df = pd.merge(df, temp, on=["breath_id","id"], how="left")

    del temp
    _ = gc.collect()

    ##############
    # Roll diffs
    ##############
    
    df["u_in_rolling_mean_lead5_diff"] = df["u_in"] - df["u_in_rolling_mean_lead5"]
    df["u_in_rolling_max5_lead5_diff"] = df["u_in"] - df["u_in_rolling_max_lead5"]
    df["u_in_rolling_min5_lead5_diff"] = df["u_in"] - df["u_in_rolling_min_lead5"]
#     df["u_in_rolling_std5_lead5_diff"] = df["u_in"] - df["u_in_rolling_std_lead5"]

    df["u_in_rolling_mean_lead10_diff"] = df["u_in"] - df["u_in_rolling_mean_lead10"]
    df["u_in_rolling_max5_lead10_diff"] = df["u_in"] - df["u_in_rolling_max_lead10"]
    df["u_in_rolling_min5_lead10_diff"] = df["u_in"] - df["u_in_rolling_min_lead10"]
#     df["u_in_rolling_std5_lead10_diff"] = df["u_in"] - df["u_in_rolling_std_lead10"]

    df["u_in_rolling_mean_lead20_diff"] = df["u_in"] - df["u_in_rolling_mean_lead20"]
    df["u_in_rolling_max5_lead20_diff"] = df["u_in"] - df["u_in_rolling_max_lead20"]
    df["u_in_rolling_min5_lead20_diff"] = df["u_in"] - df["u_in_rolling_min_lead20"]
#     df["u_in_rolling_std5_lead20_diff"] = df["u_in"] - df["u_in_rolling_std_lead20"]

    return df



def features_dummy(df):
    import cudf as pd
    import gc
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df, columns=['R','C','R__C'])
    return df



def features_group(df):
    
    import cudf as pd
    import gc

    dropCols = ['breath_steps','R__C']

    df['breath_steps'] = df[['breath_id','R']].groupby('breath_id')['R'].cumcount().reset_index(drop=True)+1
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    
    ###############
    # Resistance
    ###############
    
    print("Group R")
    
    # u_in
#     df['u_in_grp_R_mean'] = df[['R','u_in']].groupby('R').mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_max'] = df[['R','u_in']].groupby('R').max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_min'] = df[['R','u_in']].groupby('R').min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_std'] = df[['R','u_in']].groupby('R').std()["u_in"].reset_index(drop=True)

    temp1 = df[['R','u_in','u_out']].groupby(['R','u_out']).mean()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R_uout_mean'})
    temp2 = df[['R','u_in','u_out']].groupby(['R','u_out']).max()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R_uout_max'})

    temp = temp1.merge(temp2, on=['R','u_out'], how="left")
    
    df = df.merge(temp, on=['R','u_out'], how="left")

#     df['u_in_grp_R_uout_mean'] = df[['R','u_in','u_out']].groupby(['R','u_out']).mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_uout_max'] = df[['R','u_in','u_out']].groupby(['R','u_out']).max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_uout_min'] = df[['R','u_in','u_out']].groupby(['R','u_out']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_uout_std'] = df[['R','u_in','u_out']].groupby(['R','u_out']).std()["u_in"].reset_index(drop=True)

#     df['u_in_grp_R_steps_mean'] = df[['R','u_in','breath_steps']].groupby(['R','breath_steps']).mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_steps_max'] = df[['R','u_in','breath_steps']].groupby(['R','breath_steps']).max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_steps_min'] = df[['R','u_in','breath_steps']].groupby(['R','breath_steps']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R_steps_std'] = df[['R','u_in','breath_steps']].groupby(['R','breath_steps']).std()["u_in"].reset_index(drop=True)

#     df['u_in_grp_R_mean_diff'] = df["u_in"] - df["u_in_grp_R_mean"]
#     df['u_in_grp_R_max_diff'] = df["u_in"] - df["u_in_grp_R_max"]
#     df['u_in_grp_R_min_diff'] = df["u_in"] - df["u_in_grp_R_min"]
#     df['u_in_grp_R_std_diff'] = df["u_in"] - df["u_in_grp_R_std"]

    df['u_in_grp_R_uout_mean_diff'] = df["u_in"] - df["u_in_grp_R_uout_mean"]
    df['u_in_grp_R_uout_max_diff'] = df["u_in"] - df["u_in_grp_R_uout_max"]
#     df['u_in_grp_R_uout_min_diff'] = df["u_in"] - df["u_in_grp_R_uout_min"]
#     df['u_in_grp_R_uout_std_diff'] = df["u_in"] - df["u_in_grp_R_uout_std"]

#     df['u_in_grp_R_steps_mean_diff'] = df["u_in"] - df["u_in_grp_R_steps_mean"]
#     df['u_in_grp_R_steps_max_diff'] = df["u_in"] - df["u_in_grp_R_steps_max"]
#     df['u_in_grp_R_steps_min_diff'] = df["u_in"] - df["u_in_grp_R_steps_min"]
#     df['u_in_grp_R_steps_std_diff'] = df["u_in"] - df["u_in_grp_R_steps_std"]

    # area
#     df['area_grp_R_mean'] = df[['R','area']].groupby('R').mean()["area"].reset_index(drop=True)
#     df['area_grp_R_max'] = df[['R','area']].groupby('R').max()["area"].reset_index(drop=True)
#     df['area_grp_R_min'] = df[['R','area']].groupby('R').min()["area"].reset_index(drop=True)
#     df['area_grp_R_std'] = df[['R','area']].groupby('R').std()["area"].reset_index(drop=True)

    temp1 = df[['R','area','u_out']].groupby(['R','u_out']).mean()["area"].reset_index().rename(columns={'area':'area_grp_R_uout_mean'})
    temp2 = df[['R','area','u_out']].groupby(['R','u_out']).max()["area"].reset_index().rename(columns={'area':'area_grp_R_uout_max'})

    temp = temp1.merge(temp2, on=['R','u_out'], how="left")
    
    df = df.merge(temp, on=['R','u_out'], how="left")

#     df['area_grp_R_uout_min'] = df[['R','area','u_out']].groupby(['R','u_out']).min()["area"].reset_index(drop=True)
#     df['area_grp_R_uout_std'] = df[['R','area','u_out']].groupby(['R','u_out']).std()["area"].reset_index(drop=True)

#     df['area_grp_R_steps_mean'] = df[['R','area','breath_steps']].groupby(['R','breath_steps']).mean()["area"].reset_index(drop=True)
#     df['area_grp_R_steps_max'] = df[['R','area','breath_steps']].groupby(['R','breath_steps']).max()["area"].reset_index(drop=True)
#     df['area_grp_R_steps_min'] = df[['R','area','breath_steps']].groupby(['R','breath_steps']).min()["area"].reset_index(drop=True)
#     df['area_grp_R_steps_std'] = df[['R','area','breath_steps']].groupby(['R','breath_steps']).std()["area"].reset_index(drop=True)

#     df['area_grp_R_mean_diff'] = df["area"] - df["area_grp_R_mean"]
#     df['area_grp_R_max_diff'] = df["area"] - df["area_grp_R_max"]
#     df['area_grp_R_min_diff'] = df["area"] - df["area_grp_R_min"]
#     df['area_grp_R_std_diff'] = df["area"] - df["area_grp_R_std"]

    df['area_grp_R_uout_mean_diff'] = df["area"] - df["area_grp_R_uout_mean"]
    df['area_grp_R_uout_max_diff'] = df["area"] - df["area_grp_R_uout_max"]
#     df['area_grp_R_uout_min_diff'] = df["area"] - df["area_grp_R_uout_min"]
#     df['area_grp_R_uout_std_diff'] = df["area"] - df["area_grp_R_uout_std"]

#     df['area_grp_R_steps_mean_diff'] = df["area"] - df["area_grp_R_steps_mean"]
#     df['area_grp_R_steps_max_diff'] = df["area"] - df["area_grp_R_steps_max"]
#     df['area_grp_R_steps_min_diff'] = df["area"] - df["area_grp_R_steps_min"]
#     df['area_grp_R_steps_std_diff'] = df["area"] - df["area_grp_R_steps_std"]

    # u_in_sum
#     df['u_in_sum_grp_R_mean'] = df[['R','u_in_sum']].groupby('R').mean()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_max'] = df[['R','u_in_sum']].groupby('R').max()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_min'] = df[['R','u_in_sum']].groupby('R').min()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_std'] = df[['R','u_in_sum']].groupby('R').std()["u_in_sum"].reset_index(drop=True)

#     df['u_in_sum_grp_R_uout_mean'] = df[['R','u_in_sum','u_out']].groupby(['R','u_out']).mean()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_uout_max'] = df[['R','u_in_sum','u_out']].groupby(['R','u_out']).max()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_uout_min'] = df[['R','u_in_sum','u_out']].groupby(['R','u_out']).min()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R_uout_std'] = df[['R','u_in_sum','u_out']].groupby(['R','u_out']).std()["u_in_sum"].reset_index(drop=True)

#     df['u_in_sum_grp_R_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_mean"]
#     df['u_in_sum_grp_R_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_max"]
#     df['u_in_sum_grp_R_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_min"]
#     df['u_in_sum_grp_R_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_std"]

#     df['u_in_sum_grp_R_uout_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_uout_mean"]
#     df['u_in_sum_grp_R_uout_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_uout_max"]
#     df['u_in_sum_grp_R_uout_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_uout_min"]
#     df['u_in_sum_grp_R_uout_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R_uout_std"]

    # u_out_sum
#     df['u_out_sum_grp_R_mean'] = df[['R','u_out_sum']].groupby('R').mean()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_R_max'] = df[['R','u_out_sum']].groupby('R').max()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_R_min'] = df[['R','u_out_sum']].groupby('R').min()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_R_std'] = df[['R','u_out_sum']].groupby('R').std()["u_out_sum"].reset_index(drop=True)

#     df['u_out_sum_grp_R_mean_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R_mean"]
#     df['u_out_sum_grp_R_max_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R_max"]
#     df['u_out_sum_grp_R_min_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R_min"]
#     df['u_out_sum_grp_R_std_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R_std"]


    # u_in_cumsum
#     df['u_in_cumsum_grp_R_mean'] = df[['R','u_in_cumsum']].groupby('R').mean()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_max'] = df[['R','u_in_cumsum']].groupby('R').max()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_min'] = df[['R','u_in_cumsum']].groupby('R').min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_std'] = df[['R','u_in_cumsum']].groupby('R').std()["u_in_cumsum"].reset_index(drop=True)

    temp1 = df[['R','u_in_cumsum','u_out']].groupby(['R','u_out']).mean()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R_uout_mean'})
    temp2 = df[['R','u_in_cumsum','u_out']].groupby(['R','u_out']).max()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R_uout_max'})

    temp = temp1.merge(temp2, on=['R','u_out'], how="left")
    
    df = df.merge(temp, on=['R','u_out'], how="left")

#     df['u_in_cumsum_grp_R_uout_min'] = df[['R','u_in_cumsum','u_out']].groupby(['R','u_out']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_uout_std'] = df[['R','u_in_cumsum','u_out']].groupby(['R','u_out']).std()["u_in_cumsum"].reset_index(drop=True)

#     df['u_in_cumsum_grp_R_steps_mean'] = df[['R','u_in_cumsum','breath_steps']].groupby(['R','breath_steps']).mean()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_steps_max'] = df[['R','u_in_cumsum','breath_steps']].groupby(['R','breath_steps']).max()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_steps_min'] = df[['R','u_in_cumsum','breath_steps']].groupby(['R','breath_steps']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R_steps_std'] = df[['R','u_in_cumsum','breath_steps']].groupby(['R','breath_steps']).std()["u_in_cumsum"].reset_index(drop=True)

#     df['u_in_cumsum_grp_R_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_mean"]
#     df['u_in_cumsum_grp_R_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_max"]
#     df['u_in_cumsum_grp_R_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_min"]
#     df['u_in_cumsum_grp_R_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_std"]

    df['u_in_cumsum_grp_R_uout_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_uout_mean"]
    df['u_in_cumsum_grp_R_uout_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_uout_max"]
#     df['u_in_cumsum_grp_R_uout_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_uout_min"]
#     df['u_in_cumsum_grp_R_uout_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_uout_std"]

#     df['u_in_cumsum_grp_R_steps_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_steps_mean"]
#     df['u_in_cumsum_grp_R_steps_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_steps_max"]
#     df['u_in_cumsum_grp_R_steps_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_steps_min"]
#     df['u_in_cumsum_grp_R_steps_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R_steps_std"]


    # cross
#     df['cross_grp_R_mean'] = df[['R','cross']].groupby('R').mean()["cross"].reset_index(drop=True)
#     df['cross_grp_R_max'] = df[['R','cross']].groupby('R').max()["cross"].reset_index(drop=True)
#     df['cross_grp_R_min'] = df[['R','cross']].groupby('R').min()["cross"].reset_index(drop=True)
#     df['cross_grp_R_std'] = df[['R','cross']].groupby('R').std()["cross"].reset_index(drop=True)

#     df['cross_grp_R_steps_mean'] = df[['R','cross','breath_steps']].groupby(['R','breath_steps']).mean()["cross"].reset_index(drop=True)
#     df['cross_grp_R_steps_max'] = df[['R','cross','breath_steps']].groupby(['R','breath_steps']).max()["cross"].reset_index(drop=True)
#     df['cross_grp_R_steps_min'] = df[['R','cross','breath_steps']].groupby(['R','breath_steps']).min()["cross"].reset_index(drop=True)
#     df['cross_grp_R_steps_std'] = df[['R','cross','breath_steps']].groupby(['R','breath_steps']).std()["cross"].reset_index(drop=True)

#     df['cross_grp_R_mean_diff'] = df["cross"] - df["cross_grp_R_mean"]
#     df['cross_grp_R_max_diff'] = df["cross"] - df["cross_grp_R_max"]
#     df['cross_grp_R_min_diff'] = df["cross"] - df["cross_grp_R_min"]
#     df['cross_grp_R_std_diff'] = df["cross"] - df["cross_grp_R_std"]

#     df['cross_grp_R_steps_mean_diff'] = df["cross"] - df["cross_grp_R_steps_mean"]
#     df['cross_grp_R_steps_max_diff'] = df["cross"] - df["cross_grp_R_steps_max"]
#     df['cross_grp_R_steps_min_diff'] = df["cross"] - df["cross_grp_R_steps_min"]
#     df['cross_grp_R_steps_std_diff'] = df["cross"] - df["cross_grp_R_steps_std"]


    # time_passed
#     df['time_passed_grp_R_mean'] = df[['R','time_passed']].groupby('R').mean()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_max'] = df[['R','time_passed']].groupby('R').max()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_min'] = df[['R','time_passed']].groupby('R').min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_std'] = df[['R','time_passed']].groupby('R').std()["time_passed"].reset_index(drop=True)

    temp1 = df[['R','time_passed','u_out']].groupby(['R','u_out']).mean()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R_uout_mean'})
    temp2 = df[['R','time_passed','u_out']].groupby(['R','u_out']).max()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R_uout_max'})

    temp = temp1.merge(temp2, on=['R','u_out'], how="left")
    
    df = df.merge(temp, on=['R','u_out'], how="left")

#     df['time_passed_grp_R_uout_min'] = df[['R','time_passed','u_out']].groupby(['R','u_out']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_uout_std'] = df[['R','time_passed','u_out']].groupby(['R','u_out']).std()["time_passed"].reset_index(drop=True)

#     df['time_passed_grp_R_steps_mean'] = df[['R','time_passed','breath_steps']].groupby(['R','breath_steps']).mean()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_steps_max'] = df[['R','time_passed','breath_steps']].groupby(['R','breath_steps']).max()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_steps_min'] = df[['R','time_passed','breath_steps']].groupby(['R','breath_steps']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R_steps_std'] = df[['R','time_passed','breath_steps']].groupby(['R','breath_steps']).std()["time_passed"].reset_index(drop=True)

#     df['time_passed_grp_R_mean_diff'] = df["time_passed"] - df["time_passed_grp_R_mean"]
#     df['time_passed_grp_R_max_diff'] = df["time_passed"] - df["time_passed_grp_R_max"]
#     df['time_passed_grp_R_min_diff'] = df["time_passed"] - df["time_passed_grp_R_min"]
#     df['time_passed_grp_R_std_diff'] = df["time_passed"] - df["time_passed_grp_R_std"]

    df['time_passed_grp_R_uout_mean_diff'] = df["time_passed"] - df["time_passed_grp_R_uout_mean"]
    df['time_passed_grp_R_uout_max_diff'] = df["time_passed"] - df["time_passed_grp_R_uout_max"]
#     df['time_passed_grp_R_uout_min_diff'] = df["time_passed"] - df["time_passed_grp_R_uout_min"]
#     df['time_passed_grp_R_uout_std_diff'] = df["time_passed"] - df["time_passed_grp_R_uout_std"]

#     df['time_passed_grp_R_steps_mean_diff'] = df["time_passed"] - df["time_passed_grp_R_steps_mean"]
#     df['time_passed_grp_R_steps_max_diff'] = df["time_passed"] - df["time_passed_grp_R_steps_max"]
#     df['time_passed_grp_R_steps_min_diff'] = df["time_passed"] - df["time_passed_grp_R_steps_min"]
#     df['time_passed_grp_R_steps_std_diff'] = df["time_passed"] - df["time_passed_grp_R_steps_std"]
    

    ###############
    # Compliance
    ###############
    
    print("Group C")
    
    # u_in
#     df['u_in_grp_C_mean'] = df[['C','u_in']].groupby('C').mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_max'] = df[['C','u_in']].groupby('C').max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_min'] = df[['C','u_in']].groupby('C').min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_std'] = df[['C','u_in']].groupby('C').std()["u_in"].reset_index(drop=True)

    temp1 = df[['C','u_in','u_out']].groupby(['C','u_out']).mean()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_C_uout_mean'})
    temp2 = df[['C','u_in','u_out']].groupby(['C','u_out']).max()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_C_uout_max'})

    temp = temp1.merge(temp2, on=['C','u_out'], how="left")
    
    df = df.merge(temp, on=['C','u_out'], how="left")

#     df['u_in_grp_C_uout_min'] = df[['C','u_in','u_out']].groupby(['C','u_out']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_uout_std'] = df[['C','u_in','u_out']].groupby(['C','u_out']).std()["u_in"].reset_index(drop=True)

#     df['u_in_grp_C_steps_mean'] = df[['C','u_in','breath_steps']].groupby(['C','breath_steps']).mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_steps_max'] = df[['C','u_in','breath_steps']].groupby(['C','breath_steps']).max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_steps_min'] = df[['C','u_in','breath_steps']].groupby(['C','breath_steps']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_C_steps_std'] = df[['C','u_in','breath_steps']].groupby(['C','breath_steps']).std()["u_in"].reset_index(drop=True)

#     df['u_in_grp_C_mean_diff'] = df["u_in"] - df["u_in_grp_C_mean"]
#     df['u_in_grp_C_max_diff'] = df["u_in"] - df["u_in_grp_C_max"]
#     df['u_in_grp_C_min_diff'] = df["u_in"] - df["u_in_grp_C_min"]
#     df['u_in_grp_C_std_diff'] = df["u_in"] - df["u_in_grp_C_std"]

    df['u_in_grp_C_uout_mean_diff'] = df["u_in"] - df["u_in_grp_C_uout_mean"]
    df['u_in_grp_C_uout_max_diff'] = df["u_in"] - df["u_in_grp_C_uout_max"]
#     df['u_in_grp_C_uout_min_diff'] = df["u_in"] - df["u_in_grp_C_uout_min"]
#     df['u_in_grp_C_uout_std_diff'] = df["u_in"] - df["u_in_grp_C_uout_std"]

#     df['u_in_grp_C_steps_mean_diff'] = df["u_in"] - df["u_in_grp_C_steps_mean"]
#     df['u_in_grp_C_steps_max_diff'] = df["u_in"] - df["u_in_grp_C_steps_max"]
#     df['u_in_grp_C_steps_min_diff'] = df["u_in"] - df["u_in_grp_C_steps_min"]
#     df['u_in_grp_C_steps_std_diff'] = df["u_in"] - df["u_in_grp_C_steps_std"]

    # area
#     df['area_grp_C_mean'] = df[['C','area']].groupby('C').mean()["area"].reset_index(drop=True)
#     df['area_grp_C_max'] = df[['C','area']].groupby('C').max()["area"].reset_index(drop=True)
#     df['area_grp_C_min'] = df[['C','area']].groupby('C').min()["area"].reset_index(drop=True)
#     df['area_grp_C_std'] = df[['C','area']].groupby('C').std()["area"].reset_index(drop=True)

    temp1 = df[['C','area','u_out']].groupby(['C','u_out']).mean()["area"].reset_index().rename(columns={'area':'area_grp_C_uout_mean'})
    temp2 = df[['C','area','u_out']].groupby(['C','u_out']).max()["area"].reset_index().rename(columns={'area':'area_grp_C_uout_max'})

    temp = temp1.merge(temp2, on=['C','u_out'], how="left")
    
    df = df.merge(temp, on=['C','u_out'], how="left")

#     df['area_grp_C_uout_min'] = df[['C','area','u_out']].groupby(['C','u_out']).min()["area"].reset_index(drop=True)
#     df['area_grp_C_uout_std'] = df[['C','area','u_out']].groupby(['C','u_out']).std()["area"].reset_index(drop=True)

#     df['area_grp_C_steps_mean'] = df[['C','area','breath_steps']].groupby(['C','breath_steps']).mean()["area"].reset_index(drop=True)
#     df['area_grp_C_steps_max'] = df[['C','area','breath_steps']].groupby(['C','breath_steps']).max()["area"].reset_index(drop=True)
#     df['area_grp_C_steps_min'] = df[['C','area','breath_steps']].groupby(['C','breath_steps']).min()["area"].reset_index(drop=True)
#     df['area_grp_C_steps_std'] = df[['C','area','breath_steps']].groupby(['C','breath_steps']).std()["area"].reset_index(drop=True)

#     df['area_grp_C_mean_diff'] = df["area"] - df["area_grp_C_mean"]
#     df['area_grp_C_max_diff'] = df["area"] - df["area_grp_C_max"]
#     df['area_grp_C_min_diff'] = df["area"] - df["area_grp_C_min"]
#     df['area_grp_C_std_diff'] = df["area"] - df["area_grp_C_std"]

    df['area_grp_C_uout_mean_diff'] = df["area"] - df["area_grp_C_uout_mean"]
    df['area_grp_C_uout_max_diff'] = df["area"] - df["area_grp_C_uout_max"]
#     df['area_grp_C_uout_min_diff'] = df["area"] - df["area_grp_C_uout_min"]
#     df['area_grp_C_uout_std_diff'] = df["area"] - df["area_grp_C_uout_std"]

#     df['area_grp_C_steps_mean_diff'] = df["area"] - df["area_grp_C_steps_mean"]
#     df['area_grp_C_steps_max_diff'] = df["area"] - df["area_grp_C_steps_max"]
#     df['area_grp_C_steps_min_diff'] = df["area"] - df["area_grp_C_steps_min"]
#     df['area_grp_C_steps_std_diff'] = df["area"] - df["area_grp_C_steps_std"]

    # u_in_sum
#     df['u_in_sum_grp_C_mean'] = df[['C','u_in_sum']].groupby('C').mean()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_C_max'] = df[['C','u_in_sum']].groupby('C').max()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_C_min'] = df[['C','u_in_sum']].groupby('C').min()["u_in_sum"].reset_index(drop=True)
# #     df['u_in_sum_grp_C_std'] = df[['C','u_in_sum']].groupby('C').std()["u_in_sum"].reset_index(drop=True)

#     df['u_in_sum_grp_C_uout_mean'] = df[['C','u_in_sum','u_out']].groupby(['C','u_out']).mean()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_C_uout_max'] = df[['C','u_in_sum','u_out']].groupby(['C','u_out']).max()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_C_uout_min'] = df[['C','u_in_sum','u_out']].groupby(['C','u_out']).min()["u_in_sum"].reset_index(drop=True)
# #     df['u_in_sum_grp_C_uout_std'] = df[['C','u_in_sum','u_out']].groupby(['C','u_out']).std()["u_in_sum"].reset_index(drop=True)

#     df['u_in_sum_grp_C_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_mean"]
#     df['u_in_sum_grp_C_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_max"]
#     df['u_in_sum_grp_C_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_min"]
# #     df['u_in_sum_grp_C_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_std"]

#     df['u_in_sum_grp_C_uout_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_uout_mean"]
#     df['u_in_sum_grp_C_uout_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_uout_max"]
#     df['u_in_sum_grp_C_uout_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_uout_min"]
# #     df['u_in_sum_grp_C_uout_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_C_uout_std"]

#     # u_out_sum
#     df['u_out_sum_grp_C_mean'] = df[['C','u_out_sum']].groupby('C').mean()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_C_max'] = df[['C','u_out_sum']].groupby('C').max()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_C_min'] = df[['C','u_out_sum']].groupby('C').min()["u_out_sum"].reset_index(drop=True)
# #     df['u_out_sum_grp_C_std'] = df[['C','u_out_sum']].groupby('C').std()["u_out_sum"].reset_index(drop=True)

#     df['u_out_sum_grp_C_mean_diff'] = df["u_out_sum"] - df["u_out_sum_grp_C_mean"]
#     df['u_out_sum_grp_C_max_diff'] = df["u_out_sum"] - df["u_out_sum_grp_C_max"]
#     df['u_out_sum_grp_C_min_diff'] = df["u_out_sum"] - df["u_out_sum_grp_C_min"]
# #     df['u_out_sum_grp_C_std_diff'] = df["u_out_sum"] - df["u_out_sum_grp_C_std"]


    # u_in_cumsum
#     df['u_in_cumsum_grp_C_mean'] = df[['C','u_in_cumsum']].groupby('C').mean()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_max'] = df[['C','u_in_cumsum']].groupby('C').max()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_min'] = df[['C','u_in_cumsum']].groupby('C').min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_std'] = df[['C','u_in_cumsum']].groupby('C').std()["u_in_cumsum"].reset_index(drop=True)
    
    temp1 = df[['C','u_in_cumsum','u_out']].groupby(['C','u_out']).mean()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_C_uout_mean'})
    temp2 = df[['C','u_in_cumsum','u_out']].groupby(['C','u_out']).max()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_C_uout_max'})

    temp = temp1.merge(temp2, on=['C','u_out'], how="left")
    
    df = df.merge(temp, on=['C','u_out'], how="left")

#     df['u_in_cumsum_grp_C_uout_min'] = df[['C','u_in_cumsum','u_out']].groupby(['C','u_out']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_uout_std'] = df[['C','u_in_cumsum','u_out']].groupby(['C','u_out']).std()["u_in_cumsum"].reset_index(drop=True)

#     df['u_in_cumsum_grp_C_steps_mean'] = df[['C','u_in_cumsum','breath_steps']].groupby(['C','breath_steps']).mean()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_steps_max'] = df[['C','u_in_cumsum','breath_steps']].groupby(['C','breath_steps']).max()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_steps_min'] = df[['C','u_in_cumsum','breath_steps']].groupby(['C','breath_steps']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_C_steps_std'] = df[['C','u_in_cumsum','breath_steps']].groupby(['C','breath_steps']).std()["u_in_cumsum"].reset_index(drop=True)

#     df['u_in_cumsum_grp_C_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_mean"]
#     df['u_in_cumsum_grp_C_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_max"]
#     df['u_in_cumsum_grp_C_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_min"]
#     df['u_in_cumsum_grp_C_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_std"]

    df['u_in_cumsum_grp_C_uout_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_uout_mean"]
    df['u_in_cumsum_grp_C_uout_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_uout_max"]
#     df['u_in_cumsum_grp_C_uout_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_uout_min"]
#     df['u_in_cumsum_grp_C_uout_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_uout_std"]

#     df['u_in_cumsum_grp_C_steps_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_steps_mean"]
#     df['u_in_cumsum_grp_C_steps_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_steps_max"]
#     df['u_in_cumsum_grp_C_steps_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_steps_min"]
#     df['u_in_cumsum_grp_C_steps_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_C_steps_std"]


    # cross
#     df['cross_grp_C_mean'] = df[['C','cross']].groupby('C').mean()["cross"].reset_index(drop=True)
#     df['cross_grp_C_max'] = df[['C','cross']].groupby('C').max()["cross"].reset_index(drop=True)
#     df['cross_grp_C_min'] = df[['C','cross']].groupby('C').min()["cross"].reset_index(drop=True)
#     df['cross_grp_C_std'] = df[['C','cross']].groupby('C').std()["cross"].reset_index(drop=True)

#     df['cross_grp_C_steps_mean'] = df[['C','cross','breath_steps']].groupby(['C','breath_steps']).mean()["cross"].reset_index(drop=True)
#     df['cross_grp_C_steps_max'] = df[['C','cross','breath_steps']].groupby(['C','breath_steps']).max()["cross"].reset_index(drop=True)
#     df['cross_grp_C_steps_min'] = df[['C','cross','breath_steps']].groupby(['C','breath_steps']).min()["cross"].reset_index(drop=True)
#     df['cross_grp_C_steps_std'] = df[['C','cross','breath_steps']].groupby(['C','breath_steps']).std()["cross"].reset_index(drop=True)

#     df['cross_grp_C_mean_diff'] = df["cross"] - df["cross_grp_C_mean"]
#     df['cross_grp_C_max_diff'] = df["cross"] - df["cross_grp_C_max"]
#     df['cross_grp_C_min_diff'] = df["cross"] - df["cross_grp_C_min"]
#     df['cross_grp_C_std_diff'] = df["cross"] - df["cross_grp_C_std"]

#     df['cross_grp_C_steps_mean_diff'] = df["cross"] - df["cross_grp_C_steps_mean"]
#     df['cross_grp_C_steps_max_diff'] = df["cross"] - df["cross_grp_C_steps_max"]
#     df['cross_grp_C_steps_min_diff'] = df["cross"] - df["cross_grp_C_steps_min"]
#     df['cross_grp_C_steps_std_diff'] = df["cross"] - df["cross_grp_C_steps_std"]


    # time_passed
#     df['time_passed_grp_C_mean'] = df[['C','time_passed']].groupby('C').mean()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_max'] = df[['C','time_passed']].groupby('C').max()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_min'] = df[['C','time_passed']].groupby('C').min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_std'] = df[['C','time_passed']].groupby('C').std()["time_passed"].reset_index(drop=True)

    temp1 = df[['C','time_passed','u_out']].groupby(['C','u_out']).mean()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_C_uout_mean'})
    temp2 = df[['C','time_passed','u_out']].groupby(['C','u_out']).max()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_C_uout_max'})

    temp = temp1.merge(temp2, on=['C','u_out'], how="left")
    
    df = df.merge(temp, on=['C','u_out'], how="left")

#     df['time_passed_grp_C_uout_min'] = df[['C','time_passed','u_out']].groupby(['C','u_out']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_uout_std'] = df[['C','time_passed','u_out']].groupby(['C','u_out']).std()["time_passed"].reset_index(drop=True)

#     df['time_passed_grp_C_steps_mean'] = df[['C','time_passed','breath_steps']].groupby(['C','breath_steps']).mean()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_steps_max'] = df[['C','time_passed','breath_steps']].groupby(['C','breath_steps']).max()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_steps_min'] = df[['C','time_passed','breath_steps']].groupby(['C','breath_steps']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_C_steps_std'] = df[['C','time_passed','breath_steps']].groupby(['C','breath_steps']).std()["time_passed"].reset_index(drop=True)

#     df['time_passed_grp_C_mean_diff'] = df["time_passed"] - df["time_passed_grp_C_mean"]
#     df['time_passed_grp_C_max_diff'] = df["time_passed"] - df["time_passed_grp_C_max"]
#     df['time_passed_grp_C_min_diff'] = df["time_passed"] - df["time_passed_grp_C_min"]
#     df['time_passed_grp_C_std_diff'] = df["time_passed"] - df["time_passed_grp_C_std"]

    df['time_passed_grp_C_uout_mean_diff'] = df["time_passed"] - df["time_passed_grp_C_uout_mean"]
    df['time_passed_grp_C_uout_max_diff'] = df["time_passed"] - df["time_passed_grp_C_uout_max"]
#     df['time_passed_grp_C_uout_min_diff'] = df["time_passed"] - df["time_passed_grp_C_uout_min"]
#     df['time_passed_grp_C_uout_std_diff'] = df["time_passed"] - df["time_passed_grp_C_uout_std"]

#     df['time_passed_grp_C_steps_mean_diff'] = df["time_passed"] - df["time_passed_grp_C_steps_mean"]
#     df['time_passed_grp_C_steps_max_diff'] = df["time_passed"] - df["time_passed_grp_C_steps_max"]
#     df['time_passed_grp_C_steps_min_diff'] = df["time_passed"] - df["time_passed_grp_C_steps_min"]
#     df['time_passed_grp_C_steps_std_diff'] = df["time_passed"] - df["time_passed_grp_C_steps_std"]
    


    #########
    # R + C
    #########

    print("Group R+C")

    # u_in
    
    temp1 = df[['R__C','u_in']].groupby(['R__C']).mean()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R__C_mean'})
    temp2 = df[['R__C','u_in']].groupby(['R__C']).max()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['u_in_grp_R__C_min'] = df[['R__C','u_in']].groupby('R__C').min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R__C_std'] = df[['R__C','u_in']].groupby('R__C').std()["u_in"].reset_index(drop=True)

    temp1 = df[['R__C','u_in','u_out']].groupby(['R__C','u_out']).mean()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R__C_uout_mean'})
    temp2 = df[['R__C','u_in','u_out']].groupby(['R__C','u_out']).max()["u_in"].reset_index().rename(columns={'u_in':'u_in_grp_R__C_uout_max'})

    temp = temp1.merge(temp2, on=['R__C','u_out'], how="left")
    
    df = df.merge(temp, on=['R__C','u_out'], how="left")

#     df['u_in_grp_R__C_uout_min'] = df[['R__C','u_in','u_out']].groupby(['R__C','u_out']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R__C_uout_std'] = df[['R__C','u_in','u_out']].groupby(['R__C','u_out']).std()["u_in"].reset_index(drop=True)

#     df['u_in_grp_R__C_steps_mean'] = df[['R__C','u_in','breath_steps']].groupby(['R__C','breath_steps']).mean()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R__C_steps_max'] = df[['R__C','u_in','breath_steps']].groupby(['R__C','breath_steps']).max()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R__C_steps_min'] = df[['R__C','u_in','breath_steps']].groupby(['R__C','breath_steps']).min()["u_in"].reset_index(drop=True)
#     df['u_in_grp_R__C_steps_std'] = df[['R__C','u_in','breath_steps']].groupby(['R__C','breath_steps']).std()["u_in"].reset_index(drop=True)

    df['u_in_grp_R__C_mean_diff'] = df["u_in"] - df["u_in_grp_R__C_mean"]
    df['u_in_grp_R__C_max_diff'] = df["u_in"] - df["u_in_grp_R__C_max"]
#     df['u_in_grp_R__C_min_diff'] = df["u_in"] - df["u_in_grp_R__C_min"]
#     df['u_in_grp_R__C_std_diff'] = df["u_in"] - df["u_in_grp_R__C_std"]

    df['u_in_grp_R__C_uout_mean_diff'] = df["u_in"] - df["u_in_grp_R__C_uout_mean"]
    df['u_in_grp_R__C_uout_max_diff'] = df["u_in"] - df["u_in_grp_R__C_uout_max"]
#     df['u_in_grp_R__C_uout_min_diff'] = df["u_in"] - df["u_in_grp_R__C_uout_min"]
#     df['u_in_grp_R__C_uout_std_diff'] = df["u_in"] - df["u_in_grp_R__C_uout_std"]

#     df['u_in_grp_R__C_steps_mean_diff'] = df["u_in"] - df["u_in_grp_R__C_steps_mean"]
#     df['u_in_grp_R__C_steps_max_diff'] = df["u_in"] - df["u_in_grp_R__C_steps_max"]
#     df['u_in_grp_R__C_steps_min_diff'] = df["u_in"] - df["u_in_grp_R__C_steps_min"]
#     df['u_in_grp_R__C_steps_std_diff'] = df["u_in"] - df["u_in_grp_R__C_steps_std"]



    # area
    temp1 = df[['R__C','area']].groupby(['R__C']).mean()["area"].reset_index().rename(columns={'area':'area_grp_R__C_mean'})
    temp2 = df[['R__C','area']].groupby(['R__C']).max()["area"].reset_index().rename(columns={'area':'area_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['area_grp_R__C_min'] = df[['R__C','area']].groupby('R__C').min()["area"].reset_index(drop=True)
#     df['area_grp_R__C_std'] = df[['R__C','area']].groupby('R__C').std()["area"].reset_index(drop=True)

    temp1 = df[['R__C','area','u_out']].groupby(['R__C','u_out']).mean()["area"].reset_index().rename(columns={'area':'area_grp_R__C_uout_mean'})
    temp2 = df[['R__C','area','u_out']].groupby(['R__C','u_out']).max()["area"].reset_index().rename(columns={'area':'area_grp_R__C_uout_max'})

    temp = temp1.merge(temp2, on=['R__C','u_out'], how="left")
    
    df = df.merge(temp, on=['R__C','u_out'], how="left")

#     df['area_grp_R__C_uout_min'] = df[['R__C','area','u_out']].groupby(['R__C','u_out']).min()["area"].reset_index(drop=True)
#     df['area_grp_R__C_uout_std'] = df[['R__C','area','u_out']].groupby(['R__C','u_out']).std()["area"].reset_index(drop=True)

#     df['area_grp_R__C_steps_mean'] = df[['R__C','area','breath_steps']].groupby(['R__C','breath_steps']).mean()["area"].reset_index(drop=True)
#     df['area_grp_R__C_steps_max'] = df[['R__C','area','breath_steps']].groupby(['R__C','breath_steps']).max()["area"].reset_index(drop=True)
#     df['area_grp_R__C_steps_min'] = df[['R__C','area','breath_steps']].groupby(['R__C','breath_steps']).min()["area"].reset_index(drop=True)
#     df['area_grp_R__C_steps_std'] = df[['R__C','area','breath_steps']].groupby(['R__C','breath_steps']).std()["area"].reset_index(drop=True)

    df['area_grp_R__C_mean_diff'] = df["area"] - df["area_grp_R__C_mean"]
    df['area_grp_R__C_max_diff'] = df["area"] - df["area_grp_R__C_max"]
#     df['area_grp_R__C_min_diff'] = df["area"] - df["area_grp_R__C_min"]
#     df['area_grp_R__C_std_diff'] = df["area"] - df["area_grp_R__C_std"]

    df['area_grp_R__C_uout_mean_diff'] = df["area"] - df["area_grp_R__C_uout_mean"]
    df['area_grp_R__C_uout_max_diff'] = df["area"] - df["area_grp_R__C_uout_max"]
#     df['area_grp_R__C_uout_min_diff'] = df["area"] - df["area_grp_R__C_uout_min"]
#     df['area_grp_R__C_uout_std_diff'] = df["area"] - df["area_grp_R__C_uout_std"]

#     df['area_grp_R__C_steps_mean_diff'] = df["area"] - df["area_grp_R__C_steps_mean"]
#     df['area_grp_R__C_steps_max_diff'] = df["area"] - df["area_grp_R__C_steps_max"]
#     df['area_grp_R__C_steps_min_diff'] = df["area"] - df["area_grp_R__C_steps_min"]
#     df['area_grp_R__C_steps_std_diff'] = df["area"] - df["area_grp_R__C_steps_std"]

    # u_in_sum
    
    temp1 = df[['R__C','u_in_sum']].groupby(['R__C']).mean()["u_in_sum"].reset_index().rename(columns={'u_in_sum':'u_in_sum_grp_R__C_mean'})
    temp2 = df[['R__C','u_in_sum']].groupby(['R__C']).max()["u_in_sum"].reset_index().rename(columns={'u_in_sum':'u_in_sum_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['u_in_sum_grp_R__C_min'] = df[['R__C','u_in_sum']].groupby('R__C').min()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R__C_std'] = df[['R__C','u_in_sum']].groupby('R__C').std()["u_in_sum"].reset_index(drop=True)

#     df['u_in_sum_grp_R__C_uout_mean'] = df[['R__C','u_in_sum','u_out']].groupby(['R__C','u_out']).mean()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R__C_uout_max'] = df[['R__C','u_in_sum','u_out']].groupby(['R__C','u_out']).max()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R__C_uout_min'] = df[['R__C','u_in_sum','u_out']].groupby(['R__C','u_out']).min()["u_in_sum"].reset_index(drop=True)
#     df['u_in_sum_grp_R__C_uout_std'] = df[['R__C','u_in_sum','u_out']].groupby(['R__C','u_out']).std()["u_in_sum"].reset_index(drop=True)

    df['u_in_sum_grp_R__C_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_mean"]
    df['u_in_sum_grp_R__C_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_max"]
#     df['u_in_sum_grp_R__C_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_min"]
#     df['u_in_sum_grp_R__C_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_std"]

#     df['u_in_sum_grp_R__C_uout_mean_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_uout_mean"]
#     df['u_in_sum_grp_R__C_uout_max_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_uout_max"]
#     df['u_in_sum_grp_R__C_uout_min_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_uout_min"]
#     df['u_in_sum_grp_R__C_uout_std_diff'] = df["u_in_sum"] - df["u_in_sum_grp_R__C_uout_std"]

    # u_out_sum
#     df['u_out_sum_grp_R__C_mean'] = df[['R__C','u_out_sum']].groupby('R__C').mean()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_R__C_max'] = df[['R__C','u_out_sum']].groupby('R__C').max()["u_out_sum"].reset_index(drop=True)
#     df['u_out_sum_grp_R__C_min'] = df[['R__C','u_out_sum']].groupby('R__C').min()["u_out_sum"].reset_index(drop=True)
# #     df['u_out_sum_grp_R__C_std'] = df[['R__C','u_out_sum']].groupby('R__C').std()["u_out_sum"].reset_index(drop=True)

#     df['u_out_sum_grp_R__C_mean_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R__C_mean"]
#     df['u_out_sum_grp_R__C_max_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R__C_max"]
#     df['u_out_sum_grp_R__C_min_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R__C_min"]
#     df['u_out_sum_grp_R__C_std_diff'] = df["u_out_sum"] - df["u_out_sum_grp_R__C_std"]


    # u_in_cumsum
    
    temp1 = df[['R__C','u_in_cumsum']].groupby(['R__C']).mean()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R__C_mean'})
    temp2 = df[['R__C','u_in_cumsum']].groupby(['R__C']).max()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['u_in_cumsum_grp_R__C_min'] = df[['R__C','u_in_cumsum']].groupby('R__C').min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R__C_std'] = df[['R__C','u_in_cumsum']].groupby('R__C').std()["u_in_cumsum"].reset_index(drop=True)

    temp1 = df[['R__C','u_in_cumsum','u_out']].groupby(['R__C','u_out']).mean()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R__C_uout_mean'})
    temp2 = df[['R__C','u_in_cumsum','u_out']].groupby(['R__C','u_out']).max()["u_in_cumsum"].reset_index().rename(columns={'u_in_cumsum':'u_in_cumsum_grp_R__C_uout_max'})

    temp = temp1.merge(temp2, on=['R__C','u_out'], how="left")
    
    df = df.merge(temp, on=['R__C','u_out'], how="left")

#     df['u_in_cumsum_grp_R__C_uout_min'] = df[['R__C','u_in_cumsum','u_out']].groupby(['R__C','u_out']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R__C_uout_std'] = df[['R__C','u_in_cumsum','u_out']].groupby(['R__C','u_out']).std()["u_in_cumsum"].reset_index(drop=True)

#     df['u_in_cumsum_grp_R__C_steps_mean'] = df[['R__C','u_in_cumsum','breath_steps']].groupby(['R__C','breath_steps']).mean()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R__C_steps_max'] = df[['R__C','u_in_cumsum','breath_steps']].groupby(['R__C','breath_steps']).max()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R__C_steps_min'] = df[['R__C','u_in_cumsum','breath_steps']].groupby(['R__C','breath_steps']).min()["u_in_cumsum"].reset_index(drop=True)
#     df['u_in_cumsum_grp_R__C_steps_std'] = df[['R__C','u_in_cumsum','breath_steps']].groupby(['R__C','breath_steps']).std()["u_in_cumsum"].reset_index(drop=True)

    df['u_in_cumsum_grp_R__C_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_mean"]
    df['u_in_cumsum_grp_R__C_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_max"]
#     df['u_in_cumsum_grp_R__C_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_min"]
#     df['u_in_cumsum_grp_R__C_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_std"]

    df['u_in_cumsum_grp_R__C_uout_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_uout_mean"]
    df['u_in_cumsum_grp_R__C_uout_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_uout_max"]
#     df['u_in_cumsum_grp_R__C_uout_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_uout_min"]
#     df['u_in_cumsum_grp_R__C_uout_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_uout_std"]

#     df['u_in_cumsum_grp_R__C_steps_mean_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_steps_mean"]
#     df['u_in_cumsum_grp_R__C_steps_max_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_steps_max"]
#     df['u_in_cumsum_grp_R__C_steps_min_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_steps_min"]
#     df['u_in_cumsum_grp_R__C_steps_std_diff'] = df["u_in_cumsum"] - df["u_in_cumsum_grp_R__C_steps_std"]


    # cross
    temp1 = df[['R__C','cross']].groupby(['R__C']).mean()["cross"].reset_index().rename(columns={'cross':'cross_grp_R__C_mean'})
    temp2 = df[['R__C','cross']].groupby(['R__C']).max()["cross"].reset_index().rename(columns={'cross':'cross_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['cross_grp_R__C_min'] = df[['R__C','cross']].groupby('R__C').min()["cross"].reset_index(drop=True)
#     df['cross_grp_R__C_std'] = df[['R__C','cross']].groupby('R__C').std()["cross"].reset_index(drop=True)

#     df['cross_grp_R__C_steps_mean'] = df[['R__C','cross','breath_steps']].groupby(['R__C','breath_steps']).mean()["cross"].reset_index(drop=True)
#     df['cross_grp_R__C_steps_max'] = df[['R__C','cross','breath_steps']].groupby(['R__C','breath_steps']).max()["cross"].reset_index(drop=True)
#     df['cross_grp_R__C_steps_min'] = df[['R__C','cross','breath_steps']].groupby(['R__C','breath_steps']).min()["cross"].reset_index(drop=True)
#     df['cross_grp_R__C_steps_std'] = df[['R__C','cross','breath_steps']].groupby(['R__C','breath_steps']).std()["cross"].reset_index(drop=True)

    df['cross_grp_R__C_mean_diff'] = df["cross"] - df["cross_grp_R__C_mean"]
    df['cross_grp_R__C_max_diff'] = df["cross"] - df["cross_grp_R__C_max"]
#     df['cross_grp_R__C_min_diff'] = df["cross"] - df["cross_grp_R__C_min"]
#     df['cross_grp_R__C_std_diff'] = df["cross"] - df["cross_grp_R__C_std"]

#     df['cross_grp_R__C_steps_mean_diff'] = df["cross"] - df["cross_grp_R__C_steps_mean"]
#     df['cross_grp_R__C_steps_max_diff'] = df["cross"] - df["cross_grp_R__C_steps_max"]
#     df['cross_grp_R__C_steps_min_diff'] = df["cross"] - df["cross_grp_R__C_steps_min"]
#     df['cross_grp_R__C_steps_std_diff'] = df["cross"] - df["cross_grp_R__C_steps_std"]


    # time_passed
    temp1 = df[['R__C','time_passed']].groupby(['R__C']).mean()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R__C_mean'})
    temp2 = df[['R__C','time_passed']].groupby(['R__C']).max()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R__C_max'})

    temp = temp1.merge(temp2, on=['R__C'], how="left")
    
    df = df.merge(temp, on=['R__C'], how="left")
    
#     df['time_passed_grp_R__C_min'] = df[['R__C','time_passed']].groupby('R__C').min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R__C_std'] = df[['R__C','time_passed']].groupby('R__C').std()["time_passed"].reset_index(drop=True)

    temp1 = df[['R__C','time_passed','u_out']].groupby(['R__C','u_out']).mean()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R__C_uout_mean'})
    temp2 = df[['R__C','time_passed','u_out']].groupby(['R__C','u_out']).max()["time_passed"].reset_index().rename(columns={'time_passed':'time_passed_grp_R__C_uout_max'})

    temp = temp1.merge(temp2, on=['R__C','u_out'], how="left")
    
    df = df.merge(temp, on=['R__C','u_out'], how="left")

#     df['time_passed_grp_R__C_uout_min'] = df[['R__C','time_passed','u_out']].groupby(['R__C','u_out']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R__C_uout_std'] = df[['R__C','time_passed','u_out']].groupby(['R__C','u_out']).std()["time_passed"].reset_index(drop=True)

#     df['time_passed_grp_R__C_steps_mean'] = df[['R__C','time_passed','breath_steps']].groupby(['R__C','breath_steps']).mean()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R__C_steps_max'] = df[['R__C','time_passed','breath_steps']].groupby(['R__C','breath_steps']).max()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R__C_steps_min'] = df[['R__C','time_passed','breath_steps']].groupby(['R__C','breath_steps']).min()["time_passed"].reset_index(drop=True)
#     df['time_passed_grp_R__C_steps_std'] = df[['R__C','time_passed','breath_steps']].groupby(['R__C','breath_steps']).std()["time_passed"].reset_index(drop=True)

    df['time_passed_grp_R__C_mean_diff'] = df["time_passed"] - df["time_passed_grp_R__C_mean"]
    df['time_passed_grp_R__C_max_diff'] = df["time_passed"] - df["time_passed_grp_R__C_max"]
#     df['time_passed_grp_R__C_min_diff'] = df["time_passed"] - df["time_passed_grp_R__C_min"]
#     df['time_passed_grp_R__C_std_diff'] = df["time_passed"] - df["time_passed_grp_R__C_std"]

    df['time_passed_grp_R__C_uout_mean_diff'] = df["time_passed"] - df["time_passed_grp_R__C_uout_mean"]
    df['time_passed_grp_R__C_uout_max_diff'] = df["time_passed"] - df["time_passed_grp_R__C_uout_max"]
#     df['time_passed_grp_R__C_uout_min_diff'] = df["time_passed"] - df["time_passed_grp_R__C_uout_min"]
#     df['time_passed_grp_R__C_uout_std_diff'] = df["time_passed"] - df["time_passed_grp_R__C_uout_std"]

#     df['time_passed_grp_R__C_steps_mean_diff'] = df["time_passed"] - df["time_passed_grp_R__C_steps_mean"]
#     df['time_passed_grp_R__C_steps_max_diff'] = df["time_passed"] - df["time_passed_grp_R__C_steps_max"]
#     df['time_passed_grp_R__C_steps_min_diff'] = df["time_passed"] - df["time_passed_grp_R__C_steps_min"]
#     df['time_passed_grp_R__C_steps_std_diff'] = df["time_passed"] - df["time_passed_grp_R__C_steps_std"]


#     df.drop(columns=dropCols, inplace=True)
    
    return df

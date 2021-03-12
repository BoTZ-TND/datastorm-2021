import os
import sys
import yaml
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

params = yaml.safe_load(open('params.yaml'))['featurize']

np.set_printoptions(suppress=True)

if len(sys.argv) != 3:
    sys.stderr.write('Argument error. Usage:\n')
    sys.stderr.write('\tpython featurization.py data-dit-path feature-dir-path\n')
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'train.csv')
val_input = os.path.join(sys.argv[1], 'validation.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')
train_output = os.path.join(sys.argv[2], 'ks_train.csv')
val_output = os.path.join(sys.argv[2], 'ks_validation.csv')
test_output = os.path.join(sys.argv[2], 'ks_test.csv')

fs_select = params['fs_select']

col_mask = ['Gender', 'Age', 'Ethnicity', 'Educational_Level',
       'Income', 'Country_region', 'Hotel_Type', 'Adults', 'Children', 'Babies',
       'Meal_Type', 'Visted_Previously', 'Previous_Cancellations',
       'Deposit_type', 'Booking_channel', 'Required_Car_Parking',
       'Use_Promotion', 'Discount_Rate', 'Room_Rate', 'Reservation_Status']

enc_cols = ['Gender','Ethnicity', 'Educational_Level',
       'Income', 'Country_region', 'Hotel_Type','Meal_Type', 'Visted_Previously', 'Previous_Cancellations',
       'Deposit_type', 'Booking_channel', 'Required_Car_Parking', 'Use_Promotion', 'Discount_Rate', 'Reservation_Status']

def date_type_conversion( df ,columns):
    for each in columns:
        for index in df.index:
                val_in = str(df.loc[index,each]).strip().split('/')
                df.loc[index,each] = '/'.join([val_in[1],val_in[0],val_in[2]])
            #print('/'.join([check_in[1],check_in[0],check_in[2]]))

        df[each] = pd.to_datetime(df[each],dayfirst=True)
    return df

def split_date(df,columns):
    df = df.copy()
    for each in columns:
        df['dayofweek_'+each] = df[each].map(lambda x:x.dayofweek)
    return df

def cat_to_int(df, columns, enc={}):
    df = df.copy()
    if enc == {} or len(columns) > len(enc) :
        maps = enc
        for col in columns:
            if col not in maps:
                mapping = {k: i for i,k in enumerate(df.loc[:,col].unique())}
                maps[col] = mapping
            df[col] = df[col].map(maps[col])
        return df, maps
    else:
        maps = enc
        for col in columns:
            df[col] = df[col].map(maps[col])
        return df

def rfe(model, X, y):
    names=pd.DataFrame(X.columns)

    rfe_mod = RFE(estimator=model, n_features_to_select=14, step=1) #RFECV(lin_reg, step=1, cv=5) 
    myvalues=rfe_mod.fit(X,y)
    myvalues.support_
    myvalues.ranking_ 

    rankings=pd.DataFrame(myvalues.ranking_) 
    col_df = rankings.reset_index()
    ch_cols = col_df.loc[col_df.iloc[:,1]==1]['index'].values
    return ch_cols

def main():
    tr_df = pd.read_csv(train_input)
    val_df = pd.read_csv(val_input)
    ts_df = pd.read_csv(test_input)

    dy_tr_df = date_type_conversion(tr_df,['Expected_checkin','Expected_checkout', 'Booking_date'])
    dy_val_df = date_type_conversion(val_df,['Expected_checkin','Expected_checkout', 'Booking_date'])
    dy_ts_df = date_type_conversion(ts_df,['Expected_checkin','Expected_checkout', 'Booking_date'])

    sp_tr_df = split_date(dy_tr_df,['Expected_checkin'])
    sp_val_df = split_date(dy_val_df,['Expected_checkin'])
    sp_ts_df = split_date(dy_ts_df,['Expected_checkin'])

    ms_tr_df = sp_tr_df.loc[:, col_mask]
    ms_val_df = sp_val_df.loc[:, col_mask]
    ms_ts_df = sp_ts_df.loc[:, col_mask[:-1]]

    enc_tr_df, maps = cat_to_int(ms_tr_df, enc_cols,{'Reservation_Status' : {'Check-In' : 1, 'Canceled' : 2, 'No-Show': 3},})
    enc_val_df = cat_to_int(ms_val_df, enc_cols, maps)
    enc_ts_df = cat_to_int(ms_ts_df, enc_cols[:-1], maps)

    if fs_select == 1:
        ch_cols = rfe(XGBClassifier(),enc_tr_df.iloc[:, :-1], enc_tr_df.iloc[:, -1])

        fin_tr_df = enc_tr_df.iloc[:, ch_cols]
        fin_tr_df['Reservation_Status'] = enc_tr_df.iloc[:, -1]

        fin_val_df = enc_val_df.iloc[:, ch_cols]
        fin_val_df['Reservation_Status'] = enc_val_df.iloc[:, -1]

        fin_ts_df = enc_ts_df.iloc[:, ch_cols]

        fin_tr_df.to_csv(train_output, index=False)
        fin_val_df.to_csv(val_output, index=False)
        fin_ts_df.to_csv(test_output, index=False)
    else:
        enc_tr_df.to_csv(train_output, index=False)
        enc_val_df.to_csv(val_output, index=False)
        enc_ts_df.to_csv(test_output, index=False)

if __name__ == '__main__':
    main()
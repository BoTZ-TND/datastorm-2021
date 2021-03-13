import os
import sys
import yaml
import numpy as np
import pandas as pd
import datetime as dt 
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from category_encoders.wrapper import PolynomialWrapper
from category_encoders import WOEEncoder

params = yaml.safe_load(open('params.yaml'))['featurize']

np.set_printoptions(suppress=True)

if len(sys.argv) != 3:
    sys.stderr.write('Argument error. Usage:\n')
    sys.stderr.write('\tpython featurization.py data-dit-path feature-dir-path\n')
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'train.csv')
val_input = os.path.join(sys.argv[1], 'validation.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')
train_output = os.path.join(sys.argv[2], 'cba_train.csv')
val_output = os.path.join(sys.argv[2], 'cba_validation.csv')
test_output = os.path.join(sys.argv[2], 'cba_test.csv')

fs_select = params['fs_select']

col_mask = ['Gender', 'Age', 'Ethnicity', 'Educational_Level',
       'Income', 'Meal_Type', 'Visted_Previously', 'Previous_Cancellations',
       'Deposit_type', 'Required_Car_Parking',
       'Use_Promotion', 'Room_Rate','month_Expected_checkin', 'dayofweek_Expected_checkin', 'stay',
       'booking_to_checkin', 'total_participants', 'Reservation_Status']

enc_cols = ['Gender', 'Ethnicity', 'Educational_Level',
       'Income','Meal_Type', 'Visted_Previously', 'Previous_Cancellations',
       'Deposit_type', 'Required_Car_Parking','Use_Promotion']

woe_enc = WOEEncoder(cols=enc_cols)
wr_woe_enc = PolynomialWrapper(woe_enc)

def woe_enc(df, encoder, cat_cols):
    tr_db_df = encoder.transform(df[cat_cols])
    df_2 = tr_db_df.iloc[:, :len(cat_cols)]
    df_3 = tr_db_df.iloc[:, len(cat_cols):]
    df_2.columns = df_2.columns.str.rstrip('_2.0')
    df_3.columns = df_3.columns.str.rstrip('_3.0')
    avg_df = (df_2+df_3)/2

    not_cat_cols = [i for i in df.columns if i not in cat_cols]
    fin_df = avg_df.join(df.loc[:, not_cat_cols])
    return fin_df

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
        df['day_'+each] = df[each].map(lambda x:x.day)
        df['month_'+each] = df[each].map(lambda x:x.month)
        df['year_'+each] = df[each].map(lambda x:x.year)
        df['dayofweek_'+each] = df[each].map(lambda x:x.dayofweek)
    return df

def stay_period(df,
                checkin_column='Expected_checkin',
                checkout_column = 'Expected_checkout',
                stay_column = 'stay'):

     ser_num = df[checkout_column] - df[checkin_column]
     nums = ser_num.dt.days.values
     df[stay_column] = pd.Series(data=nums, index=ser_num.index)
     return df

def booking_to_checkin(df,
                       booking_column= 'Booking_date',
                       checkin_column='Expected_checkin',
                       duration_column = 'booking_to_checkin'
                       ):
    ser_num = df[checkin_column]-df[booking_column]
    nums = ser_num.dt.days.values
    df[duration_column] = pd.Series(data=nums, index=ser_num.index)
    return df

def total_num_participants(df,columns = ['Adults','Children','Babies'],column_name='total_participants'):
    df[column_name] = df[columns[0]]
    for each in columns[1:]:
        df[column_name] =df[column_name]+df[each]

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

    rfe_mod = RFE(estimator=model, n_features_to_select=10, step=1) #RFECV(lin_reg, step=1, cv=5) 
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

    st_tr_df = stay_period(sp_tr_df)
    st_val_df = stay_period(sp_val_df)
    st_ts_df = stay_period(sp_ts_df)

    bk_tr_df = booking_to_checkin(st_tr_df)
    bk_val_df = booking_to_checkin(st_val_df)
    bk_ts_df = booking_to_checkin(st_ts_df)

    tp_tr_df = total_num_participants(bk_tr_df)
    tp_val_df = total_num_participants(bk_val_df)
    tp_ts_df = total_num_participants(bk_ts_df)

    enc_tr_df = cat_to_int(tp_tr_df, ['Reservation_Status'], {'Reservation_Status' : {'Check-In' : 1, 'Canceled' : 2, 'No-Show': 3},})
    enc_val_df = cat_to_int(tp_val_df, ['Reservation_Status'], {'Reservation_Status' : {'Check-In' : 1, 'Canceled' : 2, 'No-Show': 3},})

    wr_woe_enc.fit(enc_tr_df[enc_cols], enc_tr_df['Reservation_Status'])
    db_tr_df = woe_enc(enc_tr_df, wr_woe_enc, enc_cols)
    db_val_df = woe_enc(enc_val_df, wr_woe_enc, enc_cols)
    db_ts_df = woe_enc(tp_ts_df, wr_woe_enc, enc_cols)

    ms_tr_df = db_tr_df.loc[:, col_mask]
    ms_val_df = db_val_df.loc[:, col_mask]
    ms_ts_df = db_ts_df.loc[:, ['Reservation-id']+col_mask[:-1]]

    if fs_select == 1:
        ch_cols = rfe(CatBoostClassifier(), ms_tr_df.iloc[:, :-1], ms_tr_df.iloc[:, -1])

        fin_tr_df = ms_tr_df.iloc[:, ch_cols]
        fin_tr_df['Reservation_Status'] = ms_tr_df.iloc[:, -1]

        fin_val_df = ms_val_df.iloc[:, ch_cols]
        fin_val_df['Reservation_Status'] = ms_val_df.iloc[:, -1]

        fin_ts_df = ms_ts_df.iloc[:, ch_cols]

        fin_tr_df.to_csv(train_output, index=False)
        fin_val_df.to_csv(val_output, index=False)
        fin_ts_df.to_csv(test_output, index=False)
    else:
        ms_tr_df.to_csv(train_output, index=False)
        ms_val_df.to_csv(val_output, index=False)
        ms_ts_df.to_csv(test_output, index=False)

if __name__ == '__main__':
    main()
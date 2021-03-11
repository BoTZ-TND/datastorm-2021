import os
import sys
import yaml
import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

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
    
def cat_to_int(df, columns, enc={}):
    df = df.copy()
    if enc == {}:
        maps = {}
        for col in columns:
            mapping = {k: i for i,k in enumerate(df.loc[:,col].unique())}
            df[col] = df[col].map(mapping)
            maps[col] = mapping
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

    ms_tr_df = tr_df.loc[:, col_mask]
    ms_val_df = val_df.loc[:, col_mask]
    ms_ts_df = ts_df.loc[:, col_mask[:-1]]

    enc_tr_df, maps = cat_to_int(ms_tr_df, enc_cols)
    enc_val_df = cat_to_int(ms_val_df, enc_cols, maps)
    enc_ts_df = cat_to_int(ms_ts_df, enc_cols[:-1], maps)

    if fs_select == '1':
        ch_cols = rfe(RandomForestClassifier(),enc_tr_df.iloc[:, :-1], enc_tr_df.iloc[:, -1])

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
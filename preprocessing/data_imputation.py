import numpy as np
import pandas as pd

def impute_demo(data, ffill_demo=True):
    """
    Adds age and removes year of birth, imputes sex for all data points.
    If ffill_demo is True, blood pressure and bmi are ffilled. 
    """
    df = data.copy()
    df['birthyear'] = df.birthyear.unstack(0).ffill().bfill().stack().swaplevel(0,1).groupby(['id','year']).first()
    df['age'] = df.index.get_level_values('year') - df['birthyear']
    df['sex'] = df.sex.unstack(0).ffill().bfill().stack().swaplevel(0,1).groupby(['id','year']).first()
    if ffill_demo:
        df['systolic_bp'] = df.systolic_bp.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
        df['diastolic_bp'] = df.diastolic_bp.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
        df['bmi'] = df.bmi.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
    else:
        df['systolic_bp'] = df.systolic_bp.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
        df['diastolic_bp'] = df.diastolic_bp.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
        df['bmi'] = df.bmi.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
    
    cols_to_drop = ['height', 'weight', 'birthyear']
    
    return df.drop(cols_to_drop, axis=1)

def impute_data(X_train, X_test, method, X_val=None):
    """
    Imputes train and test data, either with mean or with median.
    """ 
    train = X_train.copy()
    test = X_test.copy()
    if X_val is not None:
        val = X_val.copy()
    else:
        val = None

    cols_to_impute_demo = ['systolic_bp', 'diastolic_bp', 'bmi']
    cols_to_impute_demo = [i for i in cols_to_impute_demo if i in train.columns]
    cols_to_impute_lab = [i for i in train.columns if 'test' in i]
    cols_to_impute_icd = [i for i in train.columns if 'icd9' in i]
    cols_to_impute_med = [i for i in train.columns if 'med' in i]
    
    if method == 'mean':
        means_demo = train[cols_to_impute_demo].mean()
        means_lab = train[cols_to_impute_lab].mean()
        for col_demo in cols_to_impute_demo:
            train[col_demo] = train[col_demo].replace(np.nan, means_demo.loc[col_demo])
            test[col_demo] = test[col_demo].replace(np.nan, means_demo.loc[col_demo])
            if X_val is not None:
                val[col_demo] = val[col_demo].replace(np.nan, means_demo.loc[col_demo])
        for col_lab in cols_to_impute_lab:
            train[col_lab] = train[col_lab].replace(np.nan, means_lab.loc[col_lab])
            test[col_lab] = test[col_lab].replace(np.nan, means_lab.loc[col_lab])
            if X_val is not None:
                val[col_lab] = val[col_lab].replace(np.nan, means_lab.loc[col_lab])
    elif method == 'median':
        medians_demo = train[cols_to_impute_demo].median()
        medians_lab = train[cols_to_impute_lab].median()
        for col_demo in cols_to_impute_demo:
            train[col_demo] = train[col_demo].replace(np.nan, medians_demo.loc[col_demo])
            test[col_demo] = test[col_demo].replace(np.nan, medians_demo.loc[col_demo])
            if X_val is not None:
                val[col_demo] = val[col_demo].replace(np.nan, medians_demo.loc[col_demo])
        for col_lab in cols_to_impute_lab:
            train[col_lab] = train[col_lab].replace(np.nan, medians_lab.loc[col_lab])
            test[col_lab] = test[col_lab].replace(np.nan, medians_lab.loc[col_lab])
            if X_val is not None:
                val[col_lab] = val[col_lab].replace(np.nan, medians_lab.loc[col_lab])
    else:
        raise NameError('Type of imputation method not known. Choose either mean or median.')
    
    train[cols_to_impute_icd] = train[cols_to_impute_icd].replace(np.nan, 0.0)
    test[cols_to_impute_icd] = test[cols_to_impute_icd].replace(np.nan, 0.0)
    train[cols_to_impute_med] = train[cols_to_impute_med].replace(np.nan, 0.0)
    test[cols_to_impute_med] = test[cols_to_impute_med].replace(np.nan, 0.0)
    if X_val is not None:
        val[cols_to_impute_icd] = val[cols_to_impute_icd].replace(np.nan, 0.0)
        val[cols_to_impute_med] = val[cols_to_impute_med].replace(np.nan, 0.0)
        
    return train, test, val

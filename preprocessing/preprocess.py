import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve

from categories_dicts import bucketize_icds, test_name_to_bucket, icd9bucket_to_name, med_to_categories, med_to_class
from helper import fill_categorical_to_one_hot
from data_imputation import impute_demo

def preprocessing_icd9_df(df):
    """
    One-hot encodes ICD9 data (prefixes and buckets),
    takes only those codes which appear at least 200 times,
    adds columns with ICD9 codes of previous year.
    Input: ICD9 data.
    Output: Preprocessed ICD9 data.
    """
    df_icd9 = df.copy()
    value_counts = df_icd9.icd9_prefix.value_counts()
    to_remove = value_counts[value_counts < 5000].index
    df_icd9.icd9_prefix.replace(to_remove, np.nan, inplace=True)
    df_icd9.dropna(inplace=True)
    df_icd9['icd9_prefix'] = df_icd9['icd9_prefix'].astype(int)
    df_icd9 = fill_categorical_to_one_hot(df_icd9.drop('icd9', axis=1), {'icd9_bucket': df_icd9.icd9_bucket.unique(), 
                                                               'icd9_prefix': df_icd9.icd9_prefix.unique()})
    df_icd9 = df_icd9.groupby(df_icd9.index).sum()
    df_icd9.index = pd.MultiIndex.from_tuples(df_icd9.index)
    df_icd9.index.names = ['id', 'year']
    try:
        df_icd9.drop('icd9_prefix=250', axis=1, inplace=True)
    except:
        print('ICD9 prefix 250 already dropped.')
    try:
        df_icd9.drop('icd9_bucket=3', axis=1, inplace=True)
    except:
        print('ICD9 bucket 3 already dropped.')
        
    return df_icd9.astype(bool).astype(int)

def preprocessing_lab_df(df, 
                         ffill=True,
                         baseline=2008,
                         n_steps=5):
    """
    One hot encodes the tests that have been done
    and multiplies with the test outcome.
    Done in a loop of 5 steps in order to not overload RAM.
    Keeps only tests which were performed at least 100 times.
    Input: Lab data.
    Output: Preprocessed lab data.
    """
    df_lab = df.copy()
    value_counts = df_lab.test.value_counts()
    to_remove = value_counts[value_counts < 5000].index
    df_lab.replace(to_remove, np.nan, inplace=True)
    df_lab.dropna(inplace=True)
    ids = list(df_lab.index.get_level_values('id').unique())
    length = int(len(ids)/n_steps)
    df_lab_total = []
    for i in range(n_steps):
        #print('Preprocess chunk', i)
        if i == n_steps - 1:
            df_lab_sub = df_lab.loc[ids[length*i:]]
        else:
            df_lab_sub = df_lab.loc[ids[length*i:length*(i+1)]]
        df_lab_sub = fill_categorical_to_one_hot(df_lab_sub, {'test': df_lab_sub.test.unique()})
        df_outcome = df_lab_sub.test_outcome.copy()
        df_lab_sub.drop('test_outcome', axis=1, inplace=True)
        df_lab_sub = df_lab_sub.mul(df_outcome, axis=0)
        df_lab_sub = df_lab_sub.replace(0.0, np.nan)
        df_lab_sub = df_lab_sub.groupby(df_lab_sub.index).mean()
        df_lab_sub.index = pd.MultiIndex.from_tuples(df_lab_sub.index)
        df_lab_sub.index.names = ['id', 'year']
        df_lab_total.append(df_lab_sub)
    df_lab_total = pd.concat(df_lab_total)
    ids_baseline = df_lab_total.loc[pd.IndexSlice[:, baseline], :].index.get_level_values('id').unique()
    ids_baseline_missing = [i for i in ids if i not in ids_baseline]
    tuples_missing = [(i, baseline) for i in ids_baseline_missing]
    df_conc = pd.DataFrame(data=None, index=pd.MultiIndex.from_tuples(tuples_missing))
    df_lab_total = pd.concat([df_lab_total, df_conc]).sort_index()
    if ffill:
        df_lab_total = df_lab_total.groupby(axis=0, level='id').ffill()

    return df_lab_total

def preprocessing_med_df(df,
                         drop_antidiabetic_medications=True):
    """
    Preprocessed medication data, one-hot encodes medications.
    Input: Medication data.
    Output: Preprocessed medication data.
    """
    df_med = df.copy()
    df_med = df_med.reset_index()
    df_med = df_med[df_med.med.isin(list(med_to_categories.keys()))]
    df_med['med_id'] = df_med.med.apply(lambda x: med_to_categories[x])
    df_med = df_med[df_med.med_id.isin(list(med_to_class.keys()))]
    df_med['med_class'] = df_med.med_id.apply(lambda x: med_to_class[x])
    df_med.drop(['med', 'med_id'], inplace=True, axis=1)
    df_med = df_med.explode('med_class')
    
    df_med.reset_index(inplace=True, drop=True)
    expanded_categories = np.zeros(shape=(len(df_med),len(df_med.med_class.unique())), dtype=np.int32)
    for i, row in df_med.iterrows():
            expanded_categories[i][row['med_class']] += 1
    
    df_ext = pd.DataFrame(data = expanded_categories)
    df_ext.columns = ["(med_class = " + str(x) + ")" for x in range(len(df_med.med_class.unique()))]
    df_med = pd.concat([df_med, df_ext], axis=1)
    df_med.drop('med_class', inplace=True, axis=1)
    df_med.set_index(['id', 'year'])
    df_med = df_med.groupby(['id', 'year']).sum().astype(bool).astype(int)
    
    if drop_antidiabetic_medications:
        df_med = df_med.drop('(med_class = 2)', axis=1)
    
    return df_med

def remove_outliers(data, 
                    lower,
                    upper):
    """
    Remove data outside of lower, upper percentile ranges.
    Input: Dataframe, lower and upper bounds.
    Output: Dataframe with outliers replaced as nan."""
    cols = [i for i in data.columns if 'test' in i]
    cols = np.append(cols, ['bmi', 'systolic_bp', 'diastolic_bp'])
    df = data.copy()
    for col in cols:
        df[col].mask(~df[col].between(df[col].quantile(lower), df[col].quantile(upper)), inplace=True)
        
    return df

def family_history(df, 
                   icd):
    """
    Adds predictor family history of diabetes.
    Input: Final dataframe, ICD9 data.
    Output: Final dataframe with added predictor of family history of diabetes.
    """
    ids_family_history = list(icd[icd['icd9'] == 'V18.0'].index.get_level_values('id').unique())
    df_ids = list(df.index.get_level_values('id'))
    df['family_history'] = [1 if i in ids_family_history else 0 for i in df_ids]
    
    return df

def smoking(df, 
           icd):
    """
    Adds predictor smoking (active or former).
    Input: Final dataframe, ICD9 data.
    Output: Final dataframe with added predictor smoking.
    """
    ids_smoking = list(icd[(icd['icd9'] == '305.1') | 
                           (icd['icd9'] == 'V15.82')].loc[pd.IndexSlice[:, :2008],
                                                          :].index.get_level_values('id').unique())
    df_ids = list(df.index.get_level_values('id'))
    df['smoking'] = [1 if i in ids_smoking else 0 for i in df_ids]
    
    return df

def alcohol_abuse(df, 
                  icd):
    """
    Adds predictor alcohol abuse.
    Input: Final dataframe, ICD9 data.
    Output: Final dataframe with added predictor alcohol abuse.
    """
    alcohol_icd9 = ['291', '291.0', '291.1', '291.2', '291.3', '291.5', '291.8', '291.9', 
                    '303', '303.00', '303.90', '303.91', '303.92', '303.93', '357.5', '425.5', 
                    '535.3', '571.0', '571.1', '571.2', '571.3', 
                    'F10', 'F101', 'F102', 'F103', 'F106', 'F108', 'F109', 'Z502', 'Z714']
    ids_alcohol_abuse = list(icd[icd['icd9'].isin(alcohol_icd9)].loc[pd.IndexSlice[:, :2008], 
                                                                     :].index.get_level_values('id').unique())
    df_ids = list(df.index.get_level_values('id'))
    df['alcohol_abuse'] = [1 if i in ids_alcohol_abuse else 0 for i in df_ids]
    
    return df

def pancreas_disease(df, 
                     icd):
    """
    Adds predictor pancreas disease.
    Input: Final dataframe, ICD9 data.
    Output: Final dataframe with added predictor pancreas disease.
    """
    pancreas_disease_icd9_codes = ['751.7', '863.8', '863.9', '996.86', '577.0', '577.1', '577.2', '577.8', '577.9',
                                   '157.0', '157.1', '157.2', '157.3', '157.8', '157.9', '251.3', '251.8', '251.9']
    pancreas_disease_ids = list(icd[icd['icd9'].isin(pancreas_disease_icd9_codes)].index.get_level_values('id').unique())
    df_ids = list(df.index.get_level_values('id'))
    df['pancreas_disease'] = [1 if i in pancreas_disease_ids else 0 for i in df_ids]
    
    return df

def preprocess_data(df_lab,
                    df_demo, 
                    ids_prediabetes_onset,
                    ids_diabetes_onset,
                    df_icd=None,
                    df_med=None,
                    drop_antidiabetic_medications=True,
                    ffill_lab=True,
                    ffill_demo=True,
                    n_codes=10, 
                    n_tests=50, 
                    lower=0.001, 
                    upper=0.999,
                    outlier_removal=True,
                    baseline=2008):
    """
    Preprocesses data.
    Input: Lab data, demo data, IDs of prediabetes and diabetes patients;
    optional: ICD-9 data and medication data.
    Output: Preprocessed data for all timestamps excluding labels.
    """ 
    demo_subset = df_demo.loc[list(ids_prediabetes_onset.keys())].copy()
    lab_subset = df_lab.loc[list(ids_prediabetes_onset.keys())].copy()
    df_lab = preprocessing_lab_df(lab_subset, ffill_lab, baseline)
    tests = list(df_lab.isna().sum().sort_values(ascending=True)[:n_tests].index)
    df_lab = df_lab[tests]
    df = df_lab.join(demo_subset, how='outer')
    if isinstance(df_icd, pd.DataFrame):
        icd_subset = df_icd.loc[list(ids_prediabetes_onset.keys())].copy()
        icd_codes = list(icd_subset.icd9_prefix.value_counts().iloc[:n_codes].index)
        df_icd = preprocessing_icd9_df(icd_subset)
        df = df.join(df_icd[['icd9_prefix='+str(icd_prefix) for icd_prefix in icd_codes]])
    if isinstance(df_med, pd.DataFrame):
        med_ids = df_med.index.get_level_values('id').unique()
        med_ids = [i for i in ids_prediabetes_onset.keys() if i in med_ids]
        med_subset = df_med.loc[med_ids]
        df_med = preprocessing_med_df(med_subset, drop_antidiabetic_medications)
        df = df.join(df_med)
    df = impute_demo(df, ffill_demo)
    if outlier_removal:
        df = remove_outliers(df, lower, upper)
    
    return df

def final_preprocessing_step(data,
                             ids_diabetes_onset,
                             pred_year,
                             icd=None,
                             ffill_icd=True,
                             add_family_history=True,
                             add_smoking=True,
                             add_alcohol_abuse=True,
                             add_pancrease_disease=True,
                             add_corticosteroids=True,
                             add_somatostatins=True,
                             add_other_immunosuppressants=True,
                             baseline=2008):
    """
    Final preprocessing step selects data from baseline year and adds label depending
    on forecast horizon (pred_year).
    Input: Preprocessed data, IDs for prediabetes and diabetes patients and forecast horizon (pred_year).
    Output: Final preprocessed data including labels.
    """
    if ffill_icd:
        icd_cols = [i for i in data.columns if 'icd' in i]
        data[icd_cols] = data[icd_cols].replace(0, np.nan).groupby(axis=0, level='id').ffill().replace(np.nan, 0)
    df_final = data.loc[pd.IndexSlice[:, baseline], :].copy()
    df_final_indices = list(df_final.index.get_level_values('id').unique())
    if add_family_history:
        df_final = family_history(df_final, icd)
    if add_smoking:
        df_final = smoking(df_final, icd)
    if add_alcohol_abuse:
        df_final = alcohol_abuse(df_final, icd)
    if add_pancrease_disease:
        df_final = pancreas_disease(df_final, icd)
    if not add_corticosteroids:
        df_final = df_final.drop('(med_class = 20)', axis=1)
        df_final = df_final.drop('(med_class = 21)', axis=1)
    if not add_somatostatins:
        df_final = df_final.drop('(med_class = 22)', axis=1)
    if not add_other_immunosuppressants:
        df_final = df_final.drop('(med_class = 23)', axis=1)
        
    df_final['y'] = [1 if i in ids_diabetes_onset.keys() and ids_diabetes_onset[i] <= baseline + pred_year 
                     else 0 for i in df_final_indices]
    df_final.dropna(axis=1, how='all', inplace=True)
    
    return df_final

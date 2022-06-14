import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve

from categories_dicts import bucketize_icds, test_name_to_bucket, icd9bucket_to_name

def fill_categorical_to_one_hot(df,
                                col_dict):
    """
    This function replaces categorical variables by one-hot encoded matrizes in a pandas df
    Input: Pandas dataframe, dictionary of column to categories which should be one-hotted.
    Output: One-hot-encoded dataframe.
    """
    index = df.index
    n_cat_cols = np.sum([len(df[col].unique()) for col in df.columns if col in col_dict])
    #print('Having {} columns to fill'.format(n_cat_cols))
    fill_df = np.zeros((len(df),n_cat_cols), dtype=np.float32)
    fill_non_cat_df = []
    column_cat_names = []
    column_non_cat_names = []
    column_id = 0
    for col in df.columns:
        if col in col_dict:
            #print('One-hotting {}'.format(col))
            class_to_idx = dict(zip(col_dict[col], np.arange(len(col_dict[col]))))
            indizes = df[col].map(class_to_idx).astype(np.int32)
            fill_df[np.arange(df.shape[0]), column_id + indizes] = 1.
            column_cat_names.extend([str(col) + "=" + str(x) for x in col_dict[col]])
            column_id += len(col_dict[col])
        else:
            #print('Keeping {} as is'.format(col))
            fill_non_cat_df.append(df[col])
            column_non_cat_names.append(col)
    #print('Merging everything together')
    one_hotted_df = pd.DataFrame(data=fill_df, columns=column_cat_names)
    for col_name, col in zip(column_non_cat_names, fill_non_cat_df):
        one_hotted_df[col_name] = col.values
    one_hotted_df.index = index
    return one_hotted_df

def compute_ids_prediabetes_onset(df_lab,
                                  df_icd,
                                  baseline=2008):
    """
    Computes the IDs of patients who are diagnosed with prediabetes
    according to their HbA1c (%) values or ICD-9 code.
    Prediabetes: 5.7% <= HbA1c < 6.5% or ICD-9 code of 790.2
    Input: Lab data and ICD-9 data.
    Output: Dictionary with IDs as keys and year of diagnosis as value.
    """
    df_lab_prediabetes = df_lab[(df_lab['test'] == 'hba1c(%)') & (df_lab['test_outcome'] >= 5.7) 
                             & (df_lab['test_outcome'] < 6.5)]
    df_lab_prediabetes = df_lab_prediabetes.reset_index().groupby('id').min().year

    df_icd_prediabetes = df_icd[df_icd.icd9.apply(lambda x: '790.2' in x)].reset_index().groupby('id').min().year
    df_prediabetes = df_icd_prediabetes.to_frame().rename({'year': 'year_icd'}, axis=1).join(df_lab_prediabetes, 
                                                                                             how='outer')
    df_prediabetes = dict(df_prediabetes.min(axis=1))
    df_prediabetes = {k:df_prediabetes[k] for k in df_prediabetes.keys() if df_prediabetes[k] <= baseline}
    
    return df_prediabetes

def compute_ids_diabetes_onset(df_lab,
                               df_icd,
                               df_med,
                               antidiabetes_medication):
    """
    Computes the IDs of patients who are diagnosed with diabetes
    according to their HbA1c (%) values, ICD-9 codes or prescribed antidiabetes medication.
    Diabetes: Two measurements of HbA1c >= 6.5%, ICD-9 code of 250.xx or prescribed antidiabetes medication. 
    Input: Lab data, ICD-9 data, medication data and list of antidiabetes medications.
    Output: Dictionary with IDs as keys and year of diagnosis as value.
    """
    df_lab_diabetes = df_lab[(df_lab['test'] == 'hba1c(%)') & (df_lab['test_outcome'] >= 6.5)]
    df_lab_diabetes = df_lab_diabetes.reset_index().set_index('id')
    df_lab_diabetes = df_lab_diabetes[df_lab_diabetes.index.duplicated(keep=False)]
    df_lab_diabetes = df_lab_diabetes.groupby('id').nth(0)['year']
    
    df_icd_diabetes = df_icd[df_icd.icd9_prefix == 250].reset_index().groupby('id').min().year
    
    df_med_diabetes = df_med[df_med.med.isin(antidiabetes_medication)].reset_index().groupby('id').year.min()
    
    df_diabetes = df_icd_diabetes.to_frame().rename({'year': 'year_icd'}, axis=1).join(df_lab_diabetes, 
                                                                                              how='outer')
    df_diabetes = df_diabetes.rename({'year': 'year_lab'}, axis=1).join(df_med_diabetes, how='outer').min(axis=1)
    df_diabetes = dict(df_diabetes)
    
    return df_diabetes

def compute_max_years(df_lab,
                      df_icd,
                      df_med):
    """
    Computes the last year of available data.
    Input: Lab data, ICD-9 data and medication data.
    Output: Dictionary of IDs (keys) with final year of available data as values.
    """
    df_icd = df_icd.copy().dropna()
    max_hba1c = df_lab[df_lab['test'] == 'hba1c(%)'].reset_index().groupby('id').nth(-2).year
    max_icd = df_icd.reset_index().groupby('id').nth(-1).year
    max_med = df_med.reset_index().groupby('id').nth(-1).year
    max_years = max_med.to_frame().rename({'year': 'year_med'},
                                          axis=1).join(max_icd, 
                                                       how='outer').rename({'year': 'year_icd'}, 
                                                                           axis=1).join(max_hba1c,
                                                                                        how='outer').max(axis=1)
    return dict(max_years)

def ids_with_sex_birthyear(demo_data,
                           ids_prediabetes_onset,
                           ids_diabetes_onset):
    """
    Returns those IDs of patients for whom sex and birthyear is available.
    Input: Demo data, prediabetes onset dictionary, diabetes onset dictionary.
    Output: Dictionaries of IDs (keys) for patients with prediabetes and diabetes and onset as value.
    """
    ids_with_sex = list(demo_data.groupby(axis=0, level='id').mean().sex.dropna().index)
    ids_with_birthyear = list(demo_data.groupby(axis=0, level='id').mean().birthyear.dropna().index)
    ids_prediabetes_onset_new = {k:ids_prediabetes_onset[k] for k in ids_prediabetes_onset.keys()
                                 if k in ids_with_sex
                                 and ids_with_birthyear}
    ids_diabetes_onset_new = {k:ids_diabetes_onset[k] for k in ids_diabetes_onset.keys() if k in ids_with_sex and
                              ids_with_birthyear}
    
    return ids_prediabetes_onset_new, ids_diabetes_onset_new

def ids(df_lab,
        df_icd,
        df_med,
        df_demo,
        antidiabetes_medication,
        baseline=2008):
    """
    Computes final sample prediabetes and diabetes patients.
    Input: Lab data, ICD-9 data, medication data, demographic data and antidiabetes medications.
    Output: Dictionaries of IDs (keys) for patients with prediabetes and diabetes and onset as value.
    """
    max_years = compute_max_years(df_lab, df_icd, df_med) 
    ids_pre = compute_ids_prediabetes_onset(df_lab, df_icd)
    ids_diab = compute_ids_diabetes_onset(df_lab, df_icd, df_med, antidiabetes_medication)
    ids_pre = {k:ids_pre[k] for k in ids_pre.keys() if k not in ids_diab.keys() or ids_diab[k] > baseline}
    ids_diab = {k:ids_diab[k] for k in ids_diab.keys() if ids_diab[k] > baseline}
    ids_pre, ids_diab = ids_with_sex_birthyear(df_demo, ids_pre, ids_diab)
    ids_pre = {k:ids_pre[k] for k in ids_pre.keys() if max_years[k] == 2013}
    ids_diab = {k:ids_diab[k] for k in ids_diab.keys() if k in ids_pre.keys()}
    
    return ids_pre, ids_diab

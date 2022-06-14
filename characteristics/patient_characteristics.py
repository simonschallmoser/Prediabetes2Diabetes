import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Cm

import data_reader
import helper
import preprocess
import icd9

def add_hba1c_mmol_mol(data):
    data_new = data.copy()
    data_hba1c = data_new['test=hba1c(%)']
    data_hba1c = data_hba1c.round(1)
    for i in np.arange(3.0, 18.0, 0.1):
        data_hba1c.replace(i, np.round(10.93 * i - 23.5, 0), inplace=True)
    data_new['test=hba1c(mmol/mol)_1'] = data_hba1c
    
    return data_new   

def demo_characteristics(data_prep,
                         ids_prediabetes_onset,
                         ids_diabetes_onset,
                         ids_diabetes_icd_med,
                         ids_diabetes_lab,
                         pred_year):
    """
    Calculates demographic characteristics, returned in a dictionary.
    """
    data = preprocess.final_preprocessing_step(data_prep, ids_diabetes_onset, pred_year)
    demo_cols = [i for i in data.columns if 'test' not in i and 'icd' not in i and 'med' not in i]
    
    pre_indices = list(data[data['y'] == 0].index.get_level_values('id').unique())
    diab_indices = list(data[data['y'] == 1].index.get_level_values('id').unique())
    ids_tuple = [(i, j) for i, j in ids_diabetes_onset.items()]
    ids_data = [(i, j) for i, j in data_prep.index]
    diab_all_indices = [(i, j) for (i, j) in ids_tuple if (i, j) in ids_data]
    data_pre = data.loc[pre_indices]
    data_diab = data.loc[diab_indices]
    n_pre = len(pre_indices)
    n_diab = len(diab_indices)
    
    data_diab_all = data_prep.loc[diab_all_indices]
    data_diab_all_icd_med = data_diab_all.loc[ids_diabetes_icd_med]
    data_diab_all_lab = data_diab_all.loc[ids_diabetes_lab]
    n_diab_icd_med = len(data_diab_all_icd_med)
    n_diab_lab = len(data_diab_all_lab)
    suffix_list = ['pre', 'diab', 'diab_all', 'diab_all_icd_med', 'diab_all_lab']
    data_list = [data, data_diab, data_diab_all, data_diab_all_icd_med, data_diab_all_lab]
    n_patients_list = [len(data), n_diab, n_diab, n_diab_icd_med, n_diab_lab]
        
    demo_dict = {}
    for suffix, n_patients, data_type in zip(suffix_list, n_patients_list, data_list):
        for col in demo_cols:
            if col == 'y':
                continue
            if col == 'sex':
                if suffix == 'pre' or suffix == 'diab':
                    data_type1 = data_type
                elif suffix == 'diab_all':
                    data_type1 = data_diab
                elif suffix == 'diab_all_icd_med':
                    data_type1 = data.loc[ids_diabetes_icd_med]
                elif suffix == 'diab_all_lab':
                    data_type1 = data.loc[ids_diabetes_lab]
                demo_dict['males_'+suffix] = data_type1.sex.value_counts()[0]
                demo_dict['females_'+suffix] = data_type1.sex.value_counts()[1]
                demo_dict['males_'+suffix+'_pct'] = np.round(demo_dict['males_'+suffix]/n_patients*100, 1)
                demo_dict['females_'+suffix+'_pct'] = np.round(demo_dict['females_'+suffix]/n_patients*100, 1)
            else:
                demo_dict[col+'_'+suffix+'_mean'] = np.round(data_type[col].mean(), 1)
                demo_dict[col+'_'+suffix+'_std'] = np.round(data_type[col].std(), 1)
                if (suffix == 'pre') or (suffix == 'diab_all'):
                    demo_dict[col+'_'+suffix+'_dist'] = np.array(data_type[col])
                    
    return demo_dict

def lab_characteristics(data_prep,
                        ids_prediabetes_onset,
                        ids_diabetes_onset,
                        ids_diabetes_icd_med,
                        ids_diabetes_lab,
                        pred_year,
                        long):
    """
    Calculates biomarker characteristics, returned in a dictionary.
    """
    if not long:
        data_prep = add_hba1c_mmol_mol(data_prep)
    data = preprocess.final_preprocessing_step(data_prep, ids_diabetes_onset, pred_year)
    tests = [i.split('=')[1] for i in data.columns if 'test' in i]
    test_cols = [i for i in data.columns if 'test' in i]
    
    diab_indices = list(data[data['y'] == 1].index.get_level_values('id').unique())
    ids_tuple = [(i, j) for i, j in ids_diabetes_onset.items()]
    ids_data = [(i, j) for i, j in data_prep.index]
    diab_all_indices = [(i, j) for (i, j) in ids_tuple if (i, j) in ids_data]
    
    data_diab = data.loc[diab_indices]
    data_diab_all = data_prep.loc[diab_all_indices]
    data_diab_all_icd_med = data_diab_all.loc[ids_diabetes_icd_med]
    data_diab_all_lab = data_diab_all.loc[ids_diabetes_lab]
    suffix_list = ['pre', 'diab', 'diab_all', 'diab_all_icd_med', 'diab_all_lab']
    data_list = [data, data_diab, data_diab_all, data_diab_all_icd_med, data_diab_all_lab]
        
    lab_dict = {}
    for suffix, data_type in zip(suffix_list, data_list):
        for test, col in zip(tests, test_cols):
            if not long and (col == 'test=glucose' or col == 'test=hba1c(mmol/mol)_1'):
                round_len = 1
            else:
                round_len = 2
            lab_dict[test+'_'+suffix+'_mean'] = np.round(data_type[col].mean(), round_len)
            lab_dict[test+'_'+suffix+'_std'] = np.round(data_type[col].std(), round_len)
            if (suffix == 'pre') or (suffix == 'diab_all'):
                lab_dict[test+'_'+suffix+'_dist'] = data_type[col]
            
    return lab_dict

def icd_characteristics(data_prep,
                        ids_prediabetes_onset,
                        ids_diabetes_onset,
                        ids_diabetes_icd_med,
                        ids_diabetes_lab,
                        pred_year):
    """
    Calculates ICD-9 characteristics, returned in a dictionary.
    """
    data = preprocess.final_preprocessing_step(data_prep, ids_diabetes_onset, pred_year)
    icd_cols = [i for i in data.columns if 'icd' in i]

    diab_indices = list(data[data['y'] == 1].index.get_level_values('id').unique())
    ids_tuple = [(i, j) for i, j in ids_diabetes_onset.items()]
    ids_data = [(i, j) for i, j in data_prep.index]
    diab_all_indices = [(i, j) for (i, j) in ids_tuple if (i, j) in ids_data]
    
    data_diab = data.loc[diab_indices]
    data_icd_all = data_prep.loc[diab_all_indices]
    data_icd_all_icd_med = data_icd_all.loc[ids_diabetes_icd_med]
    data_icd_all_lab = data_icd_all.loc[ids_diabetes_lab]
    
    n_pre = len(data)
    n_diab = data.y.sum()
    n_diab_icd_med = len(data_icd_all_icd_med)
    n_diab_lab = len(data_icd_all_lab)
    
    suffix_list = ['pre', 'diab', 'diab_all', 'diab_all_icd_med', 'diab_all_lab']
    data_list = [data, data_diab, data_icd_all, data_icd_all_icd_med, data_icd_all_lab]
    n_patients_list = [n_pre, n_diab, n_diab, n_diab_icd_med, n_diab_lab]
    
    icd_dict = {}
    for suffix, n_patients, data_type in zip(suffix_list, n_patients_list, data_list):
        for col in icd_cols:
            icd_code = str(col.split('=')[1].split('_')[0])
            icd_dict[icd_code+'_'+suffix] = data_type[col].sum()
            icd_dict[icd_code+f'_{suffix}_pct'] = np.round(data_type[col].sum()/(n_patients)*100, 1)
        
    return icd_dict

def med_characteristics(data_prep,
                        ids_prediabetes_onset,
                        ids_diabetes_onset,
                        ids_diabetes_icd_med,
                        ids_diabetes_lab,
                        pred_year):
    """
    Calculates medication characteristics, returned in a dictionary.
    """
    data = preprocess.final_preprocessing_step(data_prep, ids_diabetes_onset, pred_year)
    med_cols = [i for i in data.columns if 'med' in i]
    
    diab_indices = list(data[data['y'] == 1].index.get_level_values('id').unique())
    ids_tuple = [(i, j) for i, j in ids_diabetes_onset.items()]
    ids_data = [(i, j) for i, j in data_prep.index]
    diab_all_indices = [(i, j) for (i, j) in ids_tuple if (i, j) in ids_data]
    
    data_diab = data.loc[diab_indices]
    data_med_all = data_prep.loc[diab_all_indices][med_cols]
    data_med_all_icd_med = data_med_all.loc[ids_diabetes_icd_med]
    data_med_all_lab = data_med_all.loc[ids_diabetes_lab]
    
    n_pre = len(data)
    n_diab = data.y.sum()
    n_diab_icd_med = len(data_med_all_icd_med)
    n_diab_lab = len(data_med_all_lab)
    
    suffix_list = ['pre', 'diab', 'diab_all', 'diab_all_icd_med', 'diab_all_lab']
    data_list = [data, data_diab, data_med_all, data_med_all_icd_med, data_med_all_lab]
    n_patients_list = [n_pre, n_diab, n_diab, n_diab_icd_med, n_diab_lab]
    
    med_dict = {}
    for suffix, n_patients, data_type in zip(suffix_list, n_patients_list, data_list):
        for col in med_cols:
            med_class = str(col.split('= ')[1].split(')')[0])
            med_dict[med_class+'_'+suffix] = data_type[col].sum()
            med_dict[med_class+f'_{suffix}_pct'] = np.round(data_type[col].sum()/(n_patients)*100, 1)
        
    return med_dict

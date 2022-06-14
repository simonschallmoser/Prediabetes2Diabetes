import pandas as pd
import numpy as np
from categories_dicts import bucketize_icds, test_name_to_bucket, icd9bucket_to_name, med_to_categories, med_to_class

def read_icd9_data():
    
    def digit(x):
        try:
            if x[0].isdigit():
                return int(x.split('.')[0])
        except:
            return np.nan
        else:
            return np.nan
        
    print('Read ICD9 data')
    
    df = pd.read_csv('../../../../data/INFO_AVH.csv', encoding='latin-1', 
                     header=None, usecols=[0, 1, 3])
    df['id'] = df[0]
    df['year'] = df[1].apply(lambda x: int(str(x)[:4]))
    df['icd9'] = df[3]
    #df = df[df.icd9.apply(lambda x: x[0].isdigit())]
    df['icd9'] = df.icd9.apply(lambda x: x.replace(' ',''))
    df['icd9_prefix'] = df.icd9.apply(lambda x: digit(x))
    df['icd9_prefix'] = df['icd9_prefix'].astype('Int64')
    df = df[['id','icd9','icd9_prefix','year']]
    df['icd9_bucket'] = df.icd9.apply(lambda x: bucketize_icds(x))
    
    return df.set_index(['id','year'])

def read_lab_data():
    print('Read lab data')
    df = pd.read_csv('../../../../data/INFO_LABF.csv', encoding='latin-1', header=None,
                         usecols = [0, 1, 3, 4],
                         dtype={0: 'int64', 1: 'int32', 3: str, 4: 'float32'})
    # Clean data
    df['id'] = df[0]
    df['year'] = df[1].apply(lambda x: int(str(x)[:4])).astype('int16')
    df['test'] = df[3]
    df['test_outcome'] = df[4]
    df.test = df.test.apply(lambda x: x.lower())
    df.test = df.test.apply(lambda x: x.replace('\'',''))
    df.test = df.test.apply(lambda x: x.replace(' ',''))
    df['test_outcome'] = df.test_outcome.replace(0.0, np.nan)
    df = df.dropna()
    df = df[['id', 'test', 'test_outcome', 'year']]
    df = df.set_index(['id', 'year'])

    return df

def read_demo_data():
    print('Read demo data')
    df_bmi = pd.read_csv('../../../../data/INFO_BMI.csv', encoding='latin-1', header=None)
    df_bp = pd.read_csv('../../../../data/INFO_BP.csv', encoding='latin-1', header=None)
    df_demo = pd.read_csv('../../../../data/INFO_DEMO.csv', encoding='latin-1', header=None)
    df_bmi['id'] = df_bmi[0]
    df_bmi['year'] = df_bmi[1].apply(lambda x: int(str(x)[:4])) 
    df_bmi['height'] = df_bmi[2]
    df_bmi['weight'] = df_bmi[3]
    df_bmi['bmi'] = df_bmi[4]
    df_bmi = df_bmi[['id','year','height','weight','bmi']]
    df_bmi = df_bmi.set_index(['id','year'])
    df_bp['id'] = df_bp[0]
    df_bp['year'] = df_bp[1].apply(lambda x: int(str(x)[:4])) 
    df_bp['systolic_bp'] = df_bp[2]
    df_bp['diastolic_bp'] = df_bp[3]
    df_bp = df_bp[['id','year','systolic_bp','diastolic_bp']]
    df_bp = df_bp.set_index(['id','year'])
    df_demo['id'] = df_demo[0]
    df_demo['birthyear'] = df_demo[2].apply(lambda x: int(str(x)[:4])) 
    df_demo['sex'] = df_demo[1] - 1
    df_demo = df_demo[['id','birthyear','sex']]
    df_demo = df_demo.groupby('id').first()
    
    df_ret = df_bmi.join(df_bp, how='outer')
    df_ret = df_ret.join(df_demo)   
    df_ret = df_ret.groupby(df_ret.index).mean()
    df_ret.index = pd.MultiIndex.from_tuples(df_ret.index)
    df_ret.index.names = ['id','year']
    return df_ret 

def read_med_data():
    print('Read med data')
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    def remove_amounts(s):
        return ' '.join([w for w in s.split() if not w[0].isdigit()])
    
    df_med = pd.read_csv('../../../../data/INFO_MEDC.csv', encoding='latin-1', header=None)
    df_med = df_med[[0, 2, 4]]
    df_med.columns = ['id', 'med', 'date']
    df_med = df_med[df_med.med.apply(lambda x: is_ascii(x))]
    df_med['med'] = df_med.med.apply(lambda x: remove_amounts(x))
    df_med['med'] = df_med.med.apply(lambda x: x.lower())
    df_med['med'] = df_med.med.apply(lambda x: x.replace('.', ''))
    df_med = df_med[df_med.med.apply(lambda x: True if len(x.split()) >= 1 else False)]
    df_med['med'] = df_med.med.apply(lambda x: x.split()[0]) #as i checked, the name of the meds is almost always defined by the first word
    df_med['year'] = df_med.date.apply(lambda x: int(str(x)[:4]))
    df_med.drop('date', inplace=True, axis=1)
    df_med = df_med.drop_duplicates()
    
    df_med = df_med.set_index(['id', 'year'])
    return df_med

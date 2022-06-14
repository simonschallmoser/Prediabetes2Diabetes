import numpy as np
import pandas as pd
import shap
import sys
sys.path.insert(0, '../preprocessing')

import data_imputation

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

def initialize_classifier(classifier):
    if classifier == 'logreg':
        clf = LogisticRegression(n_jobs=1, solver='saga')
    elif classifier == 'catboost':
        clf = CatBoostClassifier(verbose=0, thread_count=1)
    elif classifier == 'lgbm':
        clf = LGBMClassifier(n_jobs=1)
    return clf

def prediction(model,
               random_seed,
               pred_year,
               features=None,
               scale=False,
               imputation_method='median',
               n_iter=20,
               n_splits=5,
               n_inner_splits=4,
               sex=None):
    """
    Predicts the risk of transition from prediabetes to diabetes,
    uses nested cross-validaton to simultaneously performe hyperaparameter tuning with a randomized search 
    and to evaluate the performance on unseen patients (out-of-sample).
    """
    # Define certain variables depending on the model
    classifier = 'catboost'
    data = pd.read_csv(f'data/data_{pred_year}.csv', index_col=['id', 'year'])

    if model == 'simplified':
        features = ['test=glucose', 'bmi', 'age', 'test=hba1c(%)', 'test=creatinineserum', 
                    'test=alt(gpt)', 'test=hdl-cholest.', 'test=triglycerides']
    elif model == 'logreg_full':
        scale = True
        classifier = 'logreg'
    elif model == 'logreg_simplified':
        scale = True
        classifier = 'logreg'
        features = ['test=glucose', 'bmi', 'age', 'test=hba1c(%)', 'test=creatinineserum', 
                    'test=alt(gpt)', 'test=hdl-cholest.', 'test=triglycerides']
    elif model == 'fdrsm':
        scale = True
        classifier = 'logreg'
        features = ['test=glucose', 'bmi', 'test=hdl-cholest.', 'test=triglycerides', 
                    'systolic_bp', 'diastolic_bp', 'family_history']
    elif model == 'male':
        sex = 0
    elif model == 'female':
        sex = 1
    elif model == 'wo_glucose':
        data = data.drop('test=glucose', axis=1)
    elif model == 'wo_icd':
        icd_cols = [i for i in data.columns if 'icd' in i]
        data = data.drop(icd_cols, axis=1)
    elif model == 'wo_antidiabetic_medications':
        data = pd.read_csv(f'data/data_wo_antidiabetic_medications_{pred_year}.csv', index_col=['id', 'year'])
        
    if model != 'manual' and model != 'fdrsm':
        cols = ['family_history', 'alcohol_abuse', 'smoking', 'pancreas_disease',
                '(med_class = 20)', '(med_class = 21)', '(med_class = 22)', '(med_class = 23)']
        data = data.drop(cols, axis=1)
    if sex is not None:
        data = data[data['sex'] == sex]
        data = data.drop('sex', axis=1)

    if classifier == 'catboost':
        param_grid = {
                  'iterations': np.arange(5, 250), 
                  'learning_rate': np.logspace(-3, 0, 50),
                  'depth': np.arange(2, 9),
                  }

    elif classifier == 'logreg':
        param_grid = {
                      'C': np.logspace(-3, 3, 50),
                      'max_iter': np.arange(50, 1000),
                      }
    
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
    
    # Define X and y
    X = data.drop('y', axis=1)

    # If features are specified, select only those
    if features is not None:
        X = X[features]
    y = data['y']
    print(model, np.shape(X))
    
    # Initialize classifier
    clf = initialize_classifier(classifier)
    clf.set_params(random_state=random_seed)

    # Model prediction using stratified nested cross-validation
    y_tests = []
    y_preds = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if n_iter > 0:
            best_score = 0
            best_params = None
            for params in ParameterSampler(param_grid, n_iter, random_state=random_seed):
                
                scores = []
                skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)
                for train1_index, val_index in skf.split(X_train, y_train):
                    X_train1, X_val = X_train.iloc[train1_index], X_train.iloc[val_index]
                    y_train1, y_val = y_train.iloc[train1_index], y_train.iloc[val_index]
                    
                    X_train1, X_val, _ = data_imputation.impute_data(X_train1, X_val, imputation_method)
                    
                    pipe = Pipeline([
                            ('scaler', scaler),
                            ('model', initialize_classifier(classifier).set_params(random_state=random_seed, **params))
                            ]) 
                    pipe.fit(X_train1, y_train1)
                    y_val_pred = pipe.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, y_val_pred))
                score = np.mean(scores)    
                if score > best_score:
                    best_score = score
                    best_params = params
            pipe = Pipeline([
                        ('scaler', scaler),
                        ('model', initialize_classifier(classifier).set_params(random_state=random_seed, **best_params))
                        ])
        else:
            pipe = Pipeline([
                        ('scaler', scaler),
                        ('model', initialize_classifier(classifier).set_params(random_state=random_seed))
                        ])
        X_train, X_test, _ = data_imputation.impute_data(X_train, X_test, imputation_method)
        pipe.fit(X_train, y_train)
        y_tests.append(np.array(y_test))
        y_preds.append(pipe.predict_proba(X_test)[:, 1])
    
    results = {'y_test': y_tests, 'y_pred': y_preds}
    
    np.save(f'results/results_{model}1_{pred_year}_{random_seed}.npy', results)

    return y_tests, y_preds

def shap_final_model(model,
                     random_seed,
                     pred_year,
                     imputation_method='median',
                     n_iter=20,
                     n_inner_splits=4,
                     all_shap_values=False):
    """
    Calculates the SHAP values of the final model (one specific random seed).
    Returns which predictors remain in the final model.
    """
    # Define certain variables depending on the model
    classifier = 'catboost'
    data = pd.read_csv(f'data/data_{pred_year}.csv', index_col=['id', 'year'])
    sex = None
    if model == 'male':
        sex = 0
    elif model == 'female':
        sex = 1
    elif model == 'wo_glucose':
        data = data.drop('test=glucose', axis=1)
    elif model == 'wo_icd':
        icd_cols = [i for i in data.columns if 'icd' in i]
        data = data.drop(icd_cols, axis=1)
    elif model == 'wo_antidiabetic_medications':
        data = pd.read_csv(f'data/data_wo_antidiabetic_medications_{pred_year}.csv', index_col=['id', 'year'])

    if model != 'manual':
        cols = ['family_history', 'alcohol_abuse', 'smoking', 'pancreas_disease',
                '(med_class = 20)', '(med_class = 21)', '(med_class = 22)', '(med_class = 23)']
        data = data.drop(cols, axis=1)
    if sex is not None:
        data = data[data['sex'] == sex]
        data = data.drop('sex', axis=1)

    param_grid = {
                  'iterations': np.arange(5, 250), 
                  'learning_rate': np.logspace(-3, 0, 50),
                  'depth': np.arange(2, 9),
                  }

    
    # Define X and y
    X = data.drop('y', axis=1)
    y = data['y']
    
    # Initialize classifier
    clf = initialize_classifier(classifier)
    clf.set_params(random_state=random_seed)
    
    best_score = 0
    best_params = None
    for params in ParameterSampler(param_grid, n_iter, random_state=random_seed):

        scores = []
        skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            X_train, X_val, _ = data_imputation.impute_data(X_train, X_val, imputation_method)
            clf =  initialize_classifier(classifier).set_params(random_state=random_seed, **params)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_val_pred))
        score = np.mean(scores)    
        if score > best_score:
            best_score = score
            best_params = params
    clf =  initialize_classifier(classifier).set_params(random_state=random_seed, **best_params)
    X, _, _ = data_imputation.impute_data(X, X, imputation_method)
    clf.fit(X, y)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    np.save(f'shap_data/shap_values_{model}_{random_seed}_{pred_year}.npy', shap_values)
    if all_shap_values:
        shap_values_all = explainer(X)
        shap_values_all = shap_values_all.abs.mean(0)
        np.save(f'shap_data/values_{model}_{random_seed}_{pred_year}.npy', shap_values_all.values)
        np.save(f'shap_data/features_{model}_{random_seed}_{pred_year}.npy', shap_values_all.data)
        np.save(f'shap_data/feature_names_{model}_{random_seed}_{pred_year}.npy', shap_values_all.feature_names)
        np.save(f'shap_data/op_history_{model}_{random_seed}_{pred_year}.npy', shap_values_all.op_history)
        
    np.save(f'predictors/columns_{model}_{random_seed}_{pred_year}.npy', np.array(X.columns))
    np.save(f'predictors/feature_importances_{model}_{random_seed}_{pred_year}.npy', clf.feature_importances_)
        
    return 0
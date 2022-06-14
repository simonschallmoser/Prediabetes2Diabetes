import numpy as np
from joblib import Parallel, delayed
import prediction

if __name__ == "__main__":
    models = ['full', 'simplified', 'logreg_full', 'logreg_simplified',
              'fdrsm', 'manual', 'wo_glucose',
              'male', 'female', 'wo_icd', 'wo_antidiabetic_medications']

    pred_years = np.arange(1, 6)

    idx = []
    for pred_year in pred_years:
        for seed in [10, 1111, 2029, 9183, 194, 3781, 2989, 7141, 1132, 4208]:
            idx.append([pred_year, seed])

    for model in models:
        Parallel(n_jobs=len(idx))(delayed(prediction.prediction)(model=model,
                                                                 random_seed=id_[1],
                                                                 pred_year=id_[0],
                                                                 features=None,
                                                                 scale=False,
                                                                 imputation_method='median',
                                                                 n_iter=20,
                                                                 n_splits=5)
                                  for id_ in idx)

    models = ['full', 'simplified', 'logreg_full', 'logreg_simplified', 'fdrsm']

    pred_years = np.arange(7, 8)

    idx = []
    for pred_year in pred_years:
        for seed in [10, 1111, 2029, 9183, 194, 3781, 2989, 7141, 1132, 4208]:
            idx.append([pred_year, seed])

    for model in models:
        Parallel(n_jobs=len(idx))(delayed(prediction.prediction)(model=model,
                                                                 random_seed=id_[1],
                                                                 pred_year=id_[0],
                                                                 features=None,
                                                                 scale=False,
                                                                 imputation_method='median',
                                                                 n_iter=20,
                                                                 n_splits=5)
                                  for id_ in idx)

'''
Created on Nov 2, 2016

@author: abhijit.tomar
'''
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn import grid_search
import random
random.seed(2016)
import Helper_Tools 
import json
from sklearn.metrics import make_scorer

RMSE = make_scorer(Helper_Tools.custom_mean_squared_error, greater_is_better=False)

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(S_train, y)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def preidct(self, X):
        X = np.array(X)
        folds = list(KFold(len(X), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                S_test_i[:, j] = clf.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        clf = self.stacker
        y_pred = clf.predict(S_test)[:]
        return y_pred

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))

            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            S_test[:, i] = S_test_i.mean(1)

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        # param_grid = {
        #     'n_estimators': [100],
        #     'learning_rate': [0.45, 0.05, 0.055],
        #     'subsample': [0.72, 0.75, 0.78]
        # }
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05],
            'subsample': [0.75]
        }
        grid = grid_search.GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
        grid.fit(S_train, y)

        # a little memo
        message = 'to determine local CV score of #28'

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = grid.predict(S_test)[:]

        return y_pred
    
if __name__ == '__main__':
    start_time = time.time()
    
    X_train,y_train,X_test,id_test = Helper_Tools.generate_train_test_splits('../../resources/data/dframes/final.csv')
    
    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train.columns.tolist()))
    
    base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=12
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=12
        )
    ]
    
    ensemble = Ensemble(
        n_folds=5,
        stacker=ExtraTreesRegressor(
            random_state=2016, verbose=1
        ),
        base_models=base_models
    )

    y_pred = ensemble.fit_predict(X=X_train, y=y_train, T=X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('submission_ensemble.csv', index=False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
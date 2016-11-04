'''
Created on Oct 31, 2016

@author: abhijit.tomar
'''
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.metrics import make_scorer
from sklearn import pipeline, grid_search
import random
random.seed(2016)
import Helper_Tools 
import json

RMSE = make_scorer(Helper_Tools.custom_mean_squared_error, greater_is_better=False)

if __name__ == '__main__':
    start_time = time.time()
    
    X_train,y_train,X_test,id_test = Helper_Tools.generate_train_test_splits('../../resources/data/dframes/final.csv')

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train.columns.tolist()))

    rfr = RandomForestRegressor(n_jobs=1, random_state=2016, verbose=1)

    param_grid = {'n_estimators': [500], 'max_features': [10, 12, 14]}
    model = grid_search.GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, verbose=20, scoring=RMSE)
    #model.fit(X_train, y_train)
    clf = BaggingRegressor(rfr, n_estimators=500, max_features=14)
    print ('Fitting')
    clf.fit(X_train, y_train)
    print ('Predicting')
    '''
    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Best Params:')
    print(model.best_params_)
    with open('../../resources/data/params/rfr_params.json', 'w') as outfile:
        json.dump(model.best_params_, outfile)
    print('Best CV Score:')
    print(-model.best_score_)
    #{'n_estimators': 500, 'max_features': 14}
    '''
    y_pred = clf.predict(X_test)#model.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('submission_rfr.csv', index=False)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

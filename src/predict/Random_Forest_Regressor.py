'''
Created on Oct 31, 2016

@author: abhijit.tomar
'''
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn import grid_search
import random
random.seed(2016)
import Helper_Tools 
import json

RMSE = make_scorer(Helper_Tools.custom_mean_squared_error, greater_is_better=False)

def fit_predict_rfr():
    
    start_time = time.time()
    # Load the features/attributes
    X_train,y_train,X_test,id_test = Helper_Tools.generate_train_test_splits('../../resources/data/dframes/final.csv')

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train.columns.tolist()))
    '''
    # Initialize RandomForestRegressor
    rfr = RandomForestRegressor(n_jobs=1, random_state=2016, verbose=1)
    # Set up possible values for hyper-parameters. These would be used by GridSearch to derive optimal set of hyper-parameters
    param_grid = {'n_estimators': [500], 'max_features': [10, 12, 14, 20, 30, 50, 100]}
    # Generate optimal model using GridSearchCV
    model = grid_search.GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, verbose=20, scoring=RMSE)
    # Fit the training data on the optimal model
    print ('Fitting')
    model.fit(X_train, y_train)
    
    # Show the best parameters and save
    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Best Params:')
    print(model.best_params_)
    with open('../../resources/data/params/'+type(rfr).__name__+'_params.json', 'w') as outfile:
        json.dump(model.best_params_, outfile)
    print('Best CV Score:')
    print(model.best_score_)
    print ('Predicting')
    # Predict using the optimal model
    y_pred = model.predict(X_test)
    '''
    rfr = RandomForestRegressor(n_jobs=1, random_state=2016, max_features=100, n_estimators=500,verbose=100)
    rfr.fit(X_train, y_train)
    y_pred=rfr.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0
    
    # Save the submission
    pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('../../resources/results/'+type(rfr).__name__+'submission.csv', index=False)
    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
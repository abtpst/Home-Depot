'''
Created on Nov 10, 2016

@author: abhijit.tomar

Module for generating predictions using 
Ada Boosting
'''
import numpy as np
import json
import Helper_Tools
import time
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

def gen_ada(base_regressor,X_train,y_train,X_test):
    
    rng = np.random.RandomState(1)
    ada_rgr = AdaBoostRegressor(base_regressor,
                          n_estimators=300, random_state=rng)
    
    print('Fitting ada regressor')
    ada_rgr.fit(X_train, y_train)
    print('Predicting ada regressor')
    y_pred_ada = ada_rgr.predict(X_test)
    
    print('Returning ada predictions')
    return y_pred_ada
    
if __name__=='__main__':
    
    rfr = RandomForestRegressor()
    rfr_map = json.load(open('../../resources/data/params/'+type(rfr).__name__+'_params.json'))
    rfr = RandomForestRegressor(**rfr_map)
    
    etr = ExtraTreesRegressor()
    etr_map = json.load(open('../../resources/data/params/'+type(etr).__name__+'_params.json'))
    etr = ExtraTreesRegressor(**etr_map)
    
    gbr = GradientBoostingRegressor()
    gbr_map = json.load(open('../../resources/data/params/'+type(gbr).__name__+'_params.json'))
    gbr = GradientBoostingRegressor(**gbr_map)
    
    # Load the features/attributes
    X_train,y_train,X_test,id_test = Helper_Tools.generate_train_test_splits('../../resources/data/dframes/final.csv')

    print('Number of Features: ', len(X_train.columns.tolist()))
    
    rgr_list=[gbr,rfr,etr]
    
    for base_regressor in rgr_list:
        start_time = time.time()
        y_pred=gen_ada(base_regressor, X_train, y_train, X_test)
        for i in range(len(y_pred)):
            if y_pred[i] < 1.0:
                y_pred[i] = 1.0
            if y_pred[i] > 3.0:
                y_pred[i] = 3.0
        pd.DataFrame({'id': id_test, 'relevance': y_pred}).to_csv('../../resources/results/ada_'+type(base_regressor).__name__+'_submission.csv', index=False)
        print('Ada Boosted '+type(base_regressor).__name__+' Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
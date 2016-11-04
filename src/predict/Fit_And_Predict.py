'''
Created on Oct 13, 2016

@author: abhijit.tomar
'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

def fitPredict():
    
    df_train = pd.read_pickle('../../resources/data/dframes/train_df.pickle')
    print ('Loaded train df')
    df_test = pd.read_pickle('../../resources/data/dframes/test_df.pickle')
    print ('Loaded test df')
    
    id_test = df_test['id']
    
    y_train = df_train['relevance'].values
    X_train = df_train.drop(['id','relevance'],axis=1).values
    X_test = df_test.drop(['id','relevance'],axis=1).values
    
    rf = RandomForestRegressor(n_estimators=15, max_depth=None, random_state=0)
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    print ('Fitting')
    clf.fit(X_train, y_train)
    print ('Predicting')
    y_pred = clf.predict(X_test)
    
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../../resources/results/submission.csv',index=False)
fitPredict()
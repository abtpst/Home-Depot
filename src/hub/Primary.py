'''
Created on Oct 13, 2016

@author: abhijit.tomar

Driver module
'''

from preproc import Attribute_Features, Text_Proc_Features, Distance_Metric_Features, Similarity_Features
from predict import Random_Forest_Regressor,Extra_Trees_Regressor,Gradient_Boosting_Regressor,Ensemble_Regressor

if __name__ == '__main__':
    
    # Create features from product attributes
    Attribute_Features.generate_attribute_features()
    # Create additional features from attribute features using some text processing 
    Text_Proc_Features.generate_text_proc_features()
    # Create distance metric features
    Distance_Metric_Features.generate_distance_metric_features()
    # Create similarity metrics features
    Similarity_Features.generate_similarity_features()
    # Predict using RandomForestRegressor
    Random_Forest_Regressor.fit_predict_rfr()
    # Predict using GradientBoostingRegressor
    Gradient_Boosting_Regressor.fit_predict_gbr()
    # Predict using ExtraTreesRegressor
    Extra_Trees_Regressor.fit_predict_gbr()
    # Predict using an ensemble of the above three regressors
    Ensemble_Regressor.fit_predict_ensemble()
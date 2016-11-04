'''
Created on Oct 13, 2016

@author: abhijit.tomar
'''
from preproc import Parse
from predict import Fit_And_Predict

if __name__ == '__main__':
    Parse.cleanup()
    Fit_And_Predict.fitPredict()
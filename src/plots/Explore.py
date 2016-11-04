'''
Created on Oct 26, 2016

@author: abhijit.tomar
'''
import pandas as pd
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def majoritize(input_df):
    return input_df[input_df['majority_relevance']==1]

if __name__ == '__main__':
    
    df_all = pd.read_pickle('../../resources/data/dframes/deep_combined_df.pickle')
    print ('Loaded combined df')
    
    sns.countplot(x='relevance', data=df_all)
    
    sns.plt.show()
    
    df_all['majority_relevance'] = df_all['relevance'].map(lambda x: x in [1.0,1.33,1.67,2.0,2.33,2.67,3.0])
    
    majoritize(df_all)
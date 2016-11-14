'''
Created on Oct 30, 2016

@author: abhijit.tomar

Module for adding features for similarity metrics
'''
import pandas as pd
from nltk.metrics import edit_distance
'''
Calculate edit distance between row[tokens_col] and row[tokens_search_term]
'''
def calc_edit_distance(row, col):
    dists = [min([edit_distance(w, x) for x in row['tokens_'+col]]) for w in row['tokens_search_term']]
    if dists: 
        return (min(dists), sum(dists)) 
'''
Calculate jaccard similarity between sets A and B
'''
def jaccard(A, B):
    C = A.intersection(B)
    return float(len(C)) / (len(A) + len(B) - len(C))
'''
Find out jaccard similarity between input_df[tokens_primary_col] and input_df[tokens_col]
'''
def jaccardize(input_df, primary_col, col ):
    
    return input_df.apply(lambda x: jaccard(set(x['tokens_'+primary_col]), set(x['tokens_'+col])), axis=1)

def generate_similarity_features():
    # Load distance metric features
    df_all = pd.read_csv('../../resources/data/dframes/f_features_df.csv')
    print ('Loaded combined df')
    
    col_list = ['product_title','product_description','brand','bullet']
    # Find out the jaccard similarity between search term and each of the above columns. Add those as features
    for col in col_list:
        
        df_all['jaccard_search_term_'+col] = jaccardize(df_all, 'search_term', col)
        print('Jaccarded ',col)
        
    col_list = ['product_title','product_description']   
    '''
    For each col in col_list,
    Find out the min and avg edit distance between search term and col. Add those as features
    '''
    for col in col_list:
        
        df_all['edit_dist_search_term_'+col+'_raw'] = df_all.apply(lambda x: calc_edit_distance(x, col), axis=1)
        df_all['edit_dist_search_term_'+col+'_min'] = df_all['edit_dist_search_term_'+col+'_raw'].map(lambda x: x[0])
        df_all['edit_dist_search_term_'+col+'_avg'] = df_all['edit_dist_search_term_'+col+'_raw'].map(lambda x: x[1]) / df_all['len_search_term']
        df_all.drop(['edit_dist_search_term_'+col+'_raw'], axis=1, inplace=True)
        print ('Edit distanced ',col)
    # We will drop the following columns as they have text values and are not suitable for classification
    cols_to_drop = [
    'attr',
    'search_term',
    'product_title',
    'product_description',
    'brand',
    'bullet',
    'color',
    'material',    
    'tokens_search_term',
    'tokens_product_title',
    'tokens_product_description',
    'tokens_brand',
    'tokens_bullet'
    ]
    
    final_df = df_all.drop(cols_to_drop,axis=1)
    print('Number of Features: ', len(final_df.columns.tolist()) - 2)
    # Save the final aggregate of all of the features
    final_df.to_csv('../../resources/data/dframes/final.csv')
'''
Created on Oct 30, 2016

@author: abhijit.tomar
'''
import pandas as pd
from nltk.metrics import edit_distance

def calc_edit_distance(row, col):
    dists = [min([edit_distance(w, x) for x in row['tokens_'+col]]) for w in row['tokens_search_term']]
    if dists: 
        return (min(dists), sum(dists)) 

def jaccard(A, B):
    C = A.intersection(B)
    return float(len(C)) / (len(A) + len(B) - len(C))

def jaccardize(input_df, primary_col, col ):
    
    return input_df.apply(lambda x: jaccard(set(x['tokens_'+primary_col]), set(x['tokens_'+col])), axis=1)

if __name__ == '__main__':
    
    df_all = pd.read_pickle('../../resources/data/dframes/vectorized_combined_df.pickle')
    print ('Loaded combined df')
    
    col_list = ['product_title','product_description','brand','bullet']
    
    for col in col_list:
        
        df_all['jaccard_search_term_'+col] = jaccardize(df_all, 'search_term', col)
        print('Jaccarded ',col)
        
    col_list = ['product_title','product_description']   
    
    for col in col_list:
        
        df_all['edit_dist_search_term_'+col+'_raw'] = df_all.apply(lambda x: calc_edit_distance(x, col), axis=1)
        df_all['edit_dist_search_term_'+col+'_min'] = df_all['edit_dist_search_term_'+col+'_raw'].map(lambda x: x[0])
        df_all['edit_dist_search_term_'+col+'_avg'] = df_all['edit_dist_search_term_'+col+'_raw'].map(lambda x: x[1]) / df_all['len_search_term']
        df_all.drop(['edit_dist_search_term_'+col+'_raw'], axis=1, inplace=True)
        print ('Edit distanced ',col)
    
    df_all = pd.read_pickle('../../resources/data/dframes/similarized_combined_df.pickle')
    
    cols_to_drop = [
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
    
    final_df.to_csv('../../resources/data/dframes/final.csv')
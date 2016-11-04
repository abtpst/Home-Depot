'''
Created on Oct 30, 2016

@author: abhijit.tomar
'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def fill_cosine_sims(input_df,cv_of_search_term,secondary_col,vect):
    
    vect_mat_of_col = vect.transform(input_df[col])
    
    return [cosine_similarity(cv_of_search_term[i], vect_mat_of_col[i])[0][0] for i in range(cv_of_search_term.shape[0])]

def fill_tsvd(input_df,col,vect, vtype):
        
    vect_mat_of_col = vect.transform(input_df[col])
    vect_tsvd = tsvd.fit_transform(vect_mat_of_col)
    for i in range(vect_tsvd.shape[1]):
        input_df[col+'_'+vtype+'_tsvd_'+str(i)] = vect_tsvd[:,i]
        
if __name__ == '__main__':
    
    df_all = pd.read_pickle('../../resources/data/dframes/all_features_deep_combined_df.pickle')
    print ('Loaded combined df')
    
    cv = CountVectorizer(stop_words='english', max_features=1000)
    cv.fit(df_all['search_term']+ ' '+ df_all['product_title']+ df_all['product_description']+ ' ' + df_all['bullet'])
    print ('cv fit')
    tiv = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=1000)
    tiv.fit(df_all['search_term'] + ' ' + df_all['product_title'] + ' ' + df_all['product_description'] + ' ' + df_all['bullet'])
    print ('tiv fit')
    
    cv_of_search_term = cv.transform(df_all['search_term'])
    tiv_of_search_term = tiv.transform(df_all['search_term'])
    cos_sim_cols = ['product_title','product_description','bullet']
    
    for col in cos_sim_cols:
        
        df_all['cv_cos_sim_search_term_'+col]=fill_cosine_sims(df_all, cv_of_search_term, col, cv)
        print ('cv transformed ',col)
        df_all['tiv_cos_sim_search_term_'+col]=fill_cosine_sims(df_all, tiv_of_search_term, col, tiv)
        print ('tiv transformed ',col)
        
    col_list=['search_term','product_title','product_description','bullet']
    tsvd = TruncatedSVD(n_components=10, random_state=2016)
    for col in col_list:
        fill_tsvd(df_all,col,cv,'bow')
        print('tsvd bow for ',col)
        fill_tsvd(df_all,col,tiv,'tfidf')
        print('tsvd tfidf for ',col)
    df_all.to_pickle('../../resources/data/dframes/vectorized_combined_df.pickle')
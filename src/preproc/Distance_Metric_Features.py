'''
Created on Oct 30, 2016

@author: abhijit.tomar

Module for adding features related to various distance metrics
'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
'''
Calculate and return cosine similarity between vectors of search term and secondary_col 
with respect to the vectorizer 'vect'
'''
def fill_cosine_sims(input_df,cv_of_search_term,secondary_col,vect):
    
    vect_mat_of_col = vect.transform(input_df[secondary_col])
    
    return [cosine_similarity(cv_of_search_term[i], vect_mat_of_col[i])[0][0] for i in range(cv_of_search_term.shape[0])]
'''
For the given field, 'col' in input_df, perform truncated singular value decomposition
with respect to the vectorizer 'vect'
'''
def fill_tsvd(input_df,col,vect, vtype,tsvd):
        
    vect_mat_of_col = vect.transform(input_df[col])
    vect_tsvd = tsvd.fit_transform(vect_mat_of_col)
    for i in range(vect_tsvd.shape[1]):
        input_df[col+'_'+vtype+'_tsvd_'+str(i)] = vect_tsvd[:,i]
        
def generate_distance_metric_features():
    # Load text processed features
    df_all = pd.read_csv('../../resources/data/dframes/text_proc_features_df.csv')
    print ('Loaded combined df')
    # Initialize CountVectorizer
    cv = CountVectorizer(stop_words='english', max_features=1000)
    # Learn a vocabulary dictionary of all tokens in search term, product title, description and bullets
    cv.fit(df_all['search_term']+ ' '+ df_all['product_title']+ df_all['product_description']+ ' ' + df_all['bullet'])
    print ('cv fit')
    # Initialize TfidfVectorizer
    tiv = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=1000)
    # Learn vocabulary and idf of the words in search term, product title, description and bullets
    tiv.fit(df_all['search_term'] + ' ' + df_all['product_title'] + ' ' + df_all['product_description'] + ' ' + df_all['bullet'])
    print ('tiv fit')
    # Transform search terms to document-term matrix with counts as values
    cv_of_search_term = cv.transform(df_all['search_term'])
    # Transform search terms to document-term matrix with tf idf scores as values
    tiv_of_search_term = tiv.transform(df_all['search_term'])
    # Initialize columns for calculating cosine similarity
    cos_sim_cols = ['product_title','product_description','bullet']
    # For each column
    for col in cos_sim_cols:
        # Add cosine similarity between count vectors of search term and the column as a new feature
        df_all['cv_cos_sim_search_term_'+col]=fill_cosine_sims(df_all, cv_of_search_term, col, cv)
        print ('cv transformed ',col)
        # Add cosine similarity between tfidf vectors of search term and the column as a new feature
        df_all['tiv_cos_sim_search_term_'+col]=fill_cosine_sims(df_all, tiv_of_search_term, col, tiv)
        print ('tiv transformed ',col)
        
    col_list=['search_term','product_title','product_description','bullet']
    # For each column in the above list, perform truncated singular value decomposition for dimensionality reduction
    tsvd = TruncatedSVD(n_components=10, random_state=2016)
    for col in col_list:
        fill_tsvd(df_all,col,cv,'bow',tsvd)
        print('tsvd bow for ',col)
        fill_tsvd(df_all,col,tiv,'tfidf',tsvd)
        print('tsvd tfidf for ',col)
    # Save distance metric features. Note that these have been added on top of the text proc features
    df_all.to_csv('../../resources/data/dframes/distance_metric_features_df.csv')
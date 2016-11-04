'''
Created on Oct 13, 2016

@author: abhijit.tomar
'''
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from spellcheck import Google_Spell_Check

stemmer = SnowballStemmer('english')

def stem_and_remove_stopwords(s):
    
    #word_list = str(s).encode('utf-8').strip().lower().split()
    filtered_words = [word for word in s.lower().split() if word not in stopwords.words('english')]
    print (filtered_words)
    return " ".join([stemmer.stem(word) for word in filtered_words])
    
def str_common_word(str1, str2):
    
    x=sum(int(str2.find(word)>=0) for word in str1.split())
    print (x)
    return x
    
def cleanup():
    
    df_train = pd.read_csv('../../resources/data/train/train.csv', encoding="ISO-8859-1")
    #(74067, 5)
    print ('Read training csv')
    df_test = pd.read_csv('../../resources/data/test/test.csv', encoding="ISO-8859-1")
    #(166693, 4)
    print ('Read test csv')
    #df_attr = pd.read_csv('../../resources/data/train/attributes.csv')
    #(2044803, 3)
    #print ('Read attribute csv'
    df_pro_desc = pd.read_csv('../../resources/data/train/product_descriptions.csv')
    #(124428, 2)
    print ('Read description csv')
    
    #Marker to later split the combined df
    num_train = df_train.shape[0]
    
    #Concat training and test df 
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    
    #Join with product descriptions
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    
    #Join with attributes
    #df_all = pd.merge(df_all, df_attr, how='right', on='product_uid')
    
    for i, row in df_all.iterrows():
        current_string = row['search_term']
        if(current_string in Google_Spell_Check.spell_check_dict.keys()):
            spell_checked_string=Google_Spell_Check.spell_check_dict.get(current_string)
            print (current_string,"->",spell_checked_string)
        #df_all.set_value(i,'spell_checked'," ".join(spell_checked_string))
            df_all.loc[i,'spell_checked']=spell_checked_string
        else:
            df_all.loc[i,'spell_checked']=current_string
    #Stem 
    print ('Stemming')
    df_all['spell_checked'] = df_all['spell_checked'].map(lambda x:stem_and_remove_stopwords(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:stem_and_remove_stopwords(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:stem_and_remove_stopwords(x))
    #df_all['value'] = df_all['value'].map(lambda x:stem_and_remove_stopwords(x))
    
    #Length of query is useful for relevance
    df_all['len_of_query'] = df_all['spell_checked'].map(lambda x:len(x.split())).astype(np.int64)
    print ('Done with len_of_query')
    #Combine search terms, title, description into one field
    df_all['product_info'] = df_all['spell_checked']+"\t"+df_all['product_title']+"\t"+df_all['product_description']#+"\t"+df_all['value']
    print ('Done with product_info')
    #Take common words from search terms and title
    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    print ('Done with word_in_title')
    #Take common words from search terms and description
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    print ('Done with word_in_description')
    
    #Take common words from search terms and attribute
    #df_all['word_in_attribute'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
    
    #Drop the columns we dont need for prediction
    df_all = df_all.drop(['search_term','spell_checked','product_title','product_description','product_info'],axis=1)
    
    #Save combined dataframe
    df_all.to_pickle('../../resources/data/dframes/combined_df.pickle')
    
    #Retrieve back the training set and save
    df_train = df_all.iloc[:num_train]
    df_train.to_pickle('../../resources/data/dframes/train_df.pickle')
    
    #Retrieve back the test set and save
    df_test = df_all.iloc[num_train:]
    df_test.to_pickle('../../resources/data/dframes/test_df.pickle')
    
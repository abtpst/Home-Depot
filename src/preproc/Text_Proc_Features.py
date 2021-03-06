'''
Created on Oct 26, 2016

@author: abhijit.tomar

Module for generating features after some text processing of
attribute features
'''
import pandas as pd
import Helper_Tools 
import numpy as np
import Slicing_Aides as slicer

'''
For each col_name in col_list, stem the content
'''
def stem_func(input_df,col_name_list):
    
    for col_name in col_name_list:
        input_df[col_name] = input_df[col_name].map(lambda x: Helper_Tools.str_stem(x))

'''
For each col_name in col_list, add two new features.
input_df[tokens_col_name] is the list of tokens for input_df[col_name]
input_df[len_col_name] is the length of input_df[col_name]
'''
def tok_and_len_func(input_df,col_name_list):
    
    for col_name in col_name_list:
        input_df['tokens_'+col_name] = input_df[col_name].map(lambda x: x.split())
        input_df['len_'+col_name] = input_df[col_name].map(lambda x: len(x))
        
'''
For each col in col_list, operate on the words common with col_name and create three new features.
input_df[flag_col_name_in_col] is true if input_df[col_name] is contained within input_df[col]
input_df[num_col_name_in_col] has the number of terms common between input_df[col_name] and input_df[col]
input_df[ratio_col_name_in_col] fraction of the terms in input_df[col_name] that are present in input_df[col]
'''
def flag_count_and_ratio_func(input_df, col_name, col_list):
    
    for col in col_list:
        input_df['flag_'+col_name+'_in_'+col] = input_df.apply(lambda x: int(x[col_name] in x[col]), axis=1)
        input_df['num_'+col_name+'_in_'+col] = input_df.apply(lambda x: len(set(x['tokens_'+col_name]).intersection(set(x['tokens_'+col]))), axis=1)
        input_df['ratio_'+col_name+'_in_'+col] = input_df.apply(lambda x: x['num_'+col_name+'_in_'+col]/float(x['len_'+col_name]),axis=1)

'''
Return the word at pos in col only if pos <= length of col_name and pos <= length of col
'''
def get_word_at_pos(row,col_name,col,pos):
    if pos >= len(row['tokens_'+col_name]) or pos >= row['len_'+col]:
        return 0
    return int(row['tokens_'+col_name][pos] in row[col])

'''
For each col in col_list, find out the ith word from col_name, if it appears
'''
def calculate_ith_word(input_df, col_name, col_list, max_i):        
    
    for col in col_list:
        for i in range(max_i):
            input_df[str(i)+'th_word_in_'+col] = input_df.apply(lambda x: get_word_at_pos(x, col_name, col, i),axis=1)

'''
Flag the products that have 'prop' as an attribute
'''
def flag_if_attr_has_prop(input_df, prop_df, prop):
    
    pid_with_attr_prop = pd.unique(prop_df.product_uid.ravel())
    prop_encoder = {}
    
    for pid in pid_with_attr_prop:
        prop_encoder[pid] = 1
    
    input_df['flag_attr_has_'+prop] = input_df['product_uid'].map(lambda x: prop_encoder.get(x,0)).astype(np.float)  
           
def generate_text_proc_features():
    
    # Load attribute features
    df_all = pd.read_csv('../../resources/data/dframes/attribute_features_df.csv')
    print ('Loaded combined df')
    # Stem text in the certain columns and add it those as featuers
    print ('Stemming')
    stem_func(df_all, ['search_term','product_title','product_description','brand','bullet','color','material'])
    df_all.to_csv('../../resources/data/dframes/stemmed_combined_df.csv')
    # Tokenize and get length of certain fields and add those as features
    print ('Tokenizing and finding length of fields')
    tok_and_len_func(df_all, ['search_term','product_title','product_description','brand','bullet'])
    df_all.to_csv('../../resources/data/dframes/tok_combined_df.csv')
    # Combine search term, product title and product description into product info field
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
    # Count how many times the search_term appears in product_title
    df_all['search_in_title'] = df_all['product_info'].map(lambda x:Helper_Tools.count_word_in_sent(x.split('\t')[0],x.split('\t')[1],0))
    # Count how many times the search_term appears in product_description
    df_all['search_in_description'] = df_all['product_info'].map(lambda x:Helper_Tools.count_word_in_sent(x.split('\t')[0],x.split('\t')[2],0))
    # Determine if the last word of the search_term appears in product_title
    df_all['search_last_word_in_title'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
    # Determine if the last word of the search_term appears in product_description
    df_all['search_last_word_in_description'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
    # Determine if the first word of the search_term appears in product_title
    df_all['search_first_word_in_title'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[1]))
    # Determine if the first word of the search_term appears in product_description
    df_all['search_first_word_in_description'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[2]))
    # Count the number of common words between search_title and product_title
    df_all['common_search_and_title'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0],x.split('\t')[1]))
    # Count the number of common words between search_title and product_description
    df_all['common_search_and_description'] = df_all['product_info'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0],x.split('\t')[2]))
    # Ratio of common_search_and_title over total words in search_term
    df_all['ratio_title'] = df_all['common_search_and_title']/df_all['len_search_term']
    # Ratio of common_search_and_description over number of words in search_term
    df_all['ratio_description'] = df_all['common_search_and_description']/df_all['len_search_term']
    # Combine search_term and brand as attr
    df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
    # Count the number of common words between search_term and brand
    df_all['common_search_and_brand'] = df_all['attr'].map(lambda x:Helper_Tools.count_common_word(x.split('\t')[0],x.split('\t')[1]))
    # Ratio of common_search_and_brand over number of words in brand
    df_all['ratio_brand'] = df_all['common_search_and_brand']/df_all['len_brand']
    
    '''
    For each of the fields product_title','product_description','brand','bullet',
    
    Flag -> if search term occurs in the field
    Count -> number of words that are common between search term and the field
    Find ratios -> of the common terms between search term and the field  
    '''
    print ('Flagging, counting and ratios')
    flag_count_and_ratio_func(df_all, 'search_term', ['product_title','product_description','brand','bullet']) 
    df_all.to_csv('../../resources/data/dframes/flagged_combined_df.csv')
    ''' 
    For each of the fields product_title','product_description','brand','bullet',
    
    Find out the positions where each word in the search term appears in each field
    '''
    print ('Finding ith words')
    calculate_ith_word(df_all, 'search_term', ['product_title','product_description','bullet'],10)
    df_all.to_csv('../../resources/data/dframes/ith_combined_df.csv')
    # Convert brand names to numeric values
    print ('Encoding brands')
    brands = pd.unique(df_all.brand.ravel())
    brand_encoder = {}
    index = 1000
    
    for brand in brands:
        brand_encoder[brand] = index
        index += 10
    
    brand_encoder['no_brand'] = 500
    df_all['brand_encoded'] = df_all['brand'].map(lambda x: brand_encoder.get(x,500))
    # Convert color and material attributes to numeric values by flagging the products that have these attributes. These would be two new features
    print ('Encoding attributes')
    df_attr = pd.read_csv('../../resources/data/train/attributes.csv')
    #(2044803, 3)
    print ('Read attribute csv')
    props = ['material','color']
    
    for prop in props:
        print ('Collecting '+ prop)
        prop_df = slicer.generate_sub_attr_df(df_attr,prop,['product_uid', prop])
        flag_if_attr_has_prop(df_all,prop_df, prop)
    
    pids_with_attr = pd.unique(df_attr.product_uid.ravel())
    attr_encoder = {}
    for pid in pids_with_attr:
        attr_encoder[pid] = 1
    df_all['flag_has_attr'] = df_all['product_uid'].map(lambda x: attr_encoder.get(x, 0)).astype(np.float)
    # Save text proc features. Note that these have been added on top of the attribute features
    df_all.to_csv('../../resources/data/dframes/text_proc_features_df.csv', index=False)
'''
Created on Oct 26, 2016

@author: abhijit.tomar
'''
import pandas as pd
import Helper_Tools 
import numpy as np
import Slicing_Aides as slicer

def stem_func(input_df,col_name_list):
    
    for col_name in col_name_list:
        input_df[col_name] = input_df[col_name].map(lambda x: Helper_Tools.str_stem(x))

def tok_and_len_func(input_df,col_name_list):
    
    for col_name in col_name_list:
        input_df['tokens_'+col_name] = input_df[col_name].map(lambda x: x.split())
        input_df['len_'+col_name] = input_df[col_name].map(lambda x: len(x))
        
'''
For each col in col_list, operate on the words common with col_name
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
    print (pos,col)
    print (row['tokens_'+col_name])
    print (row[col])
    print ('Returning ',int(row['tokens_'+col_name][pos] in row[col]))
    return int(row['tokens_'+col_name][pos] in row[col])

'''
For each col in col_list, find out the ith word from col_name, if it appears
'''
def calculate_ith_word(input_df, col_name, col_list, max_i):        
    
    #for idx,row in input_df.iterrows():
    for col in col_list:
        for i in range(max_i):
            input_df[str(i)+'th_word_in_'+col] = input_df.apply(lambda x: get_word_at_pos(x, col_name, col, i),axis=1)

def flag_if_attr_has_prop(input_df, prop_df, prop):
    
    pid_with_attr_prop = pd.unique(prop_df.product_uid.ravel())
    prop_encoder = {}
    
    for pid in pid_with_attr_prop:
        prop_encoder[pid] = 1
    
    input_df['flag_attr_has_'+prop] = input_df['product_uid'].map(lambda x: prop_encoder.get(x,0)).astype(np.float)  
           
if __name__ == '__main__':
    
    df_all = pd.read_pickle('../../resources/data/dframes/deep_combined_df.pickle')
    print ('Loaded combined df')
    
    print ('Stemming')
    stem_func(df_all, ['search_term','product_title','product_description','brand','bullet','color','material'])
    #df_all.to_pickle('../../resources/data/dframes/stemmed_combined_df.pickle')
    print ('Tokenizing and finding length of fields')
    tok_and_len_func(df_all, ['search_term','product_title','product_description','brand','bullet'])
    #df_all.to_pickle('../../resources/data/dframes/tok_combined_df.pickle')
    print ('Flagging, counting and ratios')
    flag_count_and_ratio_func(df_all, 'search_term', ['product_title','product_description','brand','bullet']) 
    #df_all.to_pickle('../../resources/data/dframes/flagged_combined_df.pickle')
    print ('Finding ith words')
    calculate_ith_word(df_all, 'search_term', ['product_title','product_description','bullet'],10)
    #df_all.to_pickle('../../resources/data/dframes/ith_combined_df.pickle')
    print ('Encoding brands')
    brands = pd.unique(df_all.brand.ravel())
    brand_encoder = {}
    index = 1000
    
    for brand in brands:
        brand_encoder[brand] = index
        index += 10
    
    brand_encoder['no_brand'] = 500
    df_all['brand_encoded'] = df_all['brand'].map(lambda x: brand_encoder.get(x,500))
    
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
    
    df_all.to_pickle('../../resources/data/dframes/all_features_deep_combined_df.pickle')
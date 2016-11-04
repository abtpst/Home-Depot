'''
Created on Oct 26, 2016

@author: abhijit.tomar
'''
import pandas as pd
import numpy as np
def filter_str(df, s, col='search_term'):
    return df[df[col].str.lower().str.contains(s)]

def generate_sub_attr_df(input_df, sub_string, col_names, collecting_counts=False):
    values_of_sub_attr = dict()
    input_df['is_sub_attr'] = input_df['name'].str.lower().str.contains(sub_string).fillna(False)
    for idx,row in input_df[input_df['is_sub_attr']].iterrows():
        prod_id = row['product_uid']
        value = str(row['value'])
        if collecting_counts:
            values_of_sub_attr.setdefault(prod_id, 0)
            values_of_sub_attr[prod_id] = values_of_sub_attr[prod_id] + 1
        else:
            values_of_sub_attr.setdefault(prod_id,'')
            values_of_sub_attr[prod_id] = values_of_sub_attr[prod_id] + ' ' + value
    if collecting_counts:
        result_df = pd.DataFrame.from_dict(values_of_sub_attr, orient='index').reset_index().astype(np.float)
    else:
        result_df = pd.DataFrame.from_dict(values_of_sub_attr,orient='index').reset_index()
    
    result_df.columns = col_names
    
    return result_df

def generate_custom_sub_attr_df(input_df, sub_string, col_names, qualifier_str):
    values_of_sub_attr = dict()
    input_df['is_sub_attr'] = input_df['name'].str.lower().str.contains(sub_string)
    for idx,row in input_df[input_df['is_sub_attr']].iterrows():
        prod_id = row['product_uid']
        value = row['value']
        values_of_sub_attr.setdefault(prod_id,0)
        if qualifier_str in str(value):
            values_of_sub_attr[prod_id] = 1
    
    result_df = pd.DataFrame.from_dict(values_of_sub_attr, orient='index').reset_index().astype(np.float)
    result_df.columns = col_names
    
    return result_df

def generate_dual_sub_attr_df(input_df, dual_attr_str, col_names, dual_identifiers):
    values_of_sub_attr = dict()
    input_df['is_sub_attr'] = input_df['name'].str.lower().str.contains(dual_attr_str)
    for idx,row in input_df[input_df['is_sub_attr']].iterrows():
        prod_id = row['product_uid']
        value = row['value']
        values_of_sub_attr.setdefault(prod_id,[0,0])
        if dual_identifiers[0] in str(value):
            values_of_sub_attr[prod_id][0]=1
        if dual_identifiers[1] in str(value):
            values_of_sub_attr[prod_id][1]=1
    
    result_df = pd.DataFrame.from_dict(values_of_sub_attr,orient='index').reset_index().astype(np.float)
    result_df.columns = col_names
    
    return result_df

def replace_na(input_df, cols_and_defaults):
    
    for col_name,default_vlaue in cols_and_defaults:
        input_df[col_name].fillna(default_vlaue, inplace=True)
    
    
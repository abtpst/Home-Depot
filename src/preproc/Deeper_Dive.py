'''
Created on Oct 26, 2016

@author: abhijit.tomar
'''
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.mpl_style = 'default'

import Slicing_Aides as slicer

if __name__ == '__main__':
    df_train = pd.read_csv('../../resources/data/train/train.csv', encoding="ISO-8859-1")
    #(74067, 5)
    print ('Read training csv')
    df_test = pd.read_csv('../../resources/data/test/test.csv', encoding="ISO-8859-1")
    #(166693, 4)
    print ('Read test csv')
    df_attr = pd.read_csv('../../resources/data/train/attributes.csv')
    #(2044803, 3)
    print ('Read attribute csv')
    df_pro_desc = pd.read_csv('../../resources/data/train/product_descriptions.csv')
    #(124428, 2)
    print ('Read description csv')
    
    #Marker to later split the combined df
    num_train = df_train.shape[0]
    
    print ('Drop NA values from df_attr')
    df_attr.dropna(inplace=True)
    
    print ('Collecting Brands')
    df_brand = df_attr[df_attr.name == 'MFG Brand Name'][['product_uid', 'value']].rename(columns={'value': 'brand'})
    
    print ('Collecting Bullets')
    df_bullet = slicer.generate_sub_attr_df(df_attr,'bullet',['product_uid', 'bullet'])
    
    print ('Collecting Bullet Counts')
    df_bullet_count = slicer.generate_sub_attr_df(df_attr,'bullet',['product_uid', 'bullet_count'],True)
    
    print ('Collecting Color')
    df_color = slicer.generate_sub_attr_df(df_attr,'color',['product_uid', 'color'])
   
    print ('Collecting Material')
    df_material = slicer.generate_sub_attr_df(df_attr,'material',['product_uid', 'material'])
    
    print ('Collecting Commercial/Residential')
    df_comres = slicer.generate_dual_sub_attr_df(df_attr, 'commercial / residential', ['product_uid', 'flag_commercial','flag_residential'], ['Commercial','Residential'])
     
    print ('Collecting Indoor/Outdoor')
    df_inoutdoor = slicer.generate_dual_sub_attr_df(df_attr, 'indoor/outdoor', ['product_uid', 'flag_indoor','flag_outdoor'], ['Indoor','Outdoor'])
    
    print ('Collecting Energy Star Certified')
    df_estar = slicer.generate_custom_sub_attr_df(df_attr, 'energy star certified', ['product_uid', 'flag_estar'], 'Yes')
    
    print ('Combine all dataframes')
    print ('Join training and test sets')
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    print ('Merge Descriptions')
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    print ('Merge Brands')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    print ('Merge Bullets')
    df_all = pd.merge(df_all, df_bullet, how='left', on='product_uid')
    print ('Merge Bullet Counts')
    df_all = pd.merge(df_all, df_bullet_count, how='left', on='product_uid')
    print ('Merge Colors')
    df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
    print ('Merge Materials')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    print ('Merge Commercial/Residential')
    df_all = pd.merge(df_all, df_comres, how='left', on='product_uid')
    print ('Merge Indoor/Outdoor')
    df_all = pd.merge(df_all, df_inoutdoor, how='left', on='product_uid')
    print ('Merge Estar')
    df_all = pd.merge(df_all, df_estar, how='left', on='product_uid')
    
    print ('Fill N/A values')
    slicer.replace_na(df_all, [('brand','nobrand'),('bullet',''),('bullet_count',0),('color',''),('material',''),
                               ('flag_commercial',-1),('flag_residential',-1),('flag_indoor',-1),('flag_outdoor',-1),
                               ('flag_estar',-1)])
    
    df_all.to_pickle('../../resources/data/dframes/deep_combined_df.pickle')
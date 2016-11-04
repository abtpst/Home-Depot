'''
Created on Oct 14, 2016

@author: abhijit.tomar
'''
import pandas as pd
from autocorrect import spell

if __name__ == '__main__':
    
    df_train = pd.read_csv('../../resources/data/train/train.csv', encoding="ISO-8859-1")
    #(74067, 5)
    print 'Read training csv'
    for i, row in df_train.iterrows():
        current_string = row['search_term']
        spell_checked_string=[]
        for word in current_string.split():
            spell_checked_string.append(spell(word))
        df_train.set_value(i,'spell_checked'," ".join(spell_checked_string))
    print df_train['spell_checked']
    '''    
        for word in st.split() :
            if word.isalpha():
                word=word.lower()
                pc = spell(word)
                pc = pc.lower()
                if word!=pc:
                    print word, '->' ,pc
                    c=c+1
    print "Total misspells ",c
    
    df_test = pd.read_csv('../../resources/data/test/test.csv', encoding="ISO-8859-1")
    #(166693, 4)
    print 'Read test csv'
    
    df_pro_desc = pd.read_csv('../../resources/data/train/product_descriptions.csv')
    #(124428, 2)
    print 'Read description csv'
    
    df_attr = pd.read_csv('../../resources/data/train/attributes.csv')
    #(2044803, 3)
    print 'Read attribute csv'
    #print df_attr.head(10)
    
    all_df = pd.concat([df_train,df_test],axis=0)
    print all_df.columns.values
    
    print df_attr.columns.values
    aa = pd.merge(all_df,df_attr, how='inner',on=['product_uid'])
    print 'Common between all and attr'
    
    
    da = pd.merge(df_pro_desc,df_attr, how='inner',on=['product_uid','name'])
    print 'Common between desc and attr'
    print da.head(10)
    '''
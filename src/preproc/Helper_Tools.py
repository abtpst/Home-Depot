'''
Created on Oct 26, 2016

@author: abhijit.tomar

Some methods to aid in text processing and
other common tasks
'''
import warnings
warnings.filterwarnings("ignore")
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import random
import re
random.seed(2016)

stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku', 'with', 'what', 'from', 'that', 'less', 'er', 'ing'] 
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
'''
Method for stemming. Basically removing unnecessary characters
'''
def str_stem(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("¡ã"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r
'''
Method for finding the number of common words.
if word_or_sent is a sentence (has spaces), then we return the number 
of common words between word_or_sent and sent.
Else, return 1 if word_or_sent appears in sent
'''
def count_common_word(word_or_sent, sent):
    words, cnt = word_or_sent.split(), 0
    for word in words:
        if sent.find(word)>=0:
            cnt+=1
    return cnt
'''
Method for counting the number of times word appears in sent,
beginning at starting_index
'''
def count_word_in_sent(word, sent, starting_index):
    cnt = 0
    while starting_index < len(sent):
        starting_index = sent.find(word, starting_index)
        if starting_index == -1:
            return cnt
        else:
            cnt += 1
            starting_index += len(word)
    return cnt
    
'''
Method for fixing typos in search term
by referencing Google
'''
from spellcheck.Google_Spell_Check import spell_check_dict as typo_dict
def correct_typo(s):
    if s in typo_dict:
        return typo_dict[s]
    else:
        return s
'''
Method for finding root mean squared error
'''
from sklearn.metrics import mean_squared_error
def custom_mean_squared_error(ground_truth, predictions):
    
    return mean_squared_error(ground_truth, predictions) ** 0.5
'''
Method for splitting the combined data back into training and test sets.
This is done after all of the features/attributes have been generated for 
the combined (training + test) data set
'''
import pandas as pd
def generate_train_test_splits(path_to_df):
    # Load training data
    df_train = pd.read_csv('../../resources/data/train/train.csv', encoding="ISO-8859-1")
    # Marker to later split the combined df
    num_train = df_train.shape[0]
    df_all = pd.read_csv(path_to_df, encoding='ISO-8859-1', index_col=0)
    # Slice based on marker
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]
    # Id column for test data
    id_test = df_test['id']
    # Prediction values from training data
    y_train = df_train['relevance'].values
    # The following columns are not needed as we have generated all of the features
    cols_to_drop = ['id', 'product_uid', 'relevance', 'search_term', 'product_title', 'product_description', 
                    'brand', 'attr', 'product_info',
                    'Unnamed: 0',  'Unnamed: 0.1',  'Unnamed: 0.1.1',  'Unnamed: 0.1.1']
    for col in cols_to_drop:
        try:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)
        except:
            continue
    
    # New names for df_train and df_test
    X_train = df_train[:]
    X_test = df_test[:]
    
    return X_train,y_train,X_test,id_test
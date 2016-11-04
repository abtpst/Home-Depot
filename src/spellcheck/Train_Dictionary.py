'''
Created on Oct 14, 2016

@author: abhijit.tomar
'''
import re
from collections import defaultdict

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
    
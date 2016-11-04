'''
Created on Oct 14, 2016

@author: abhijit.tomar
'''
from spellcheck import Train_Dictionary as td
from autocorrect import spell

Trained_WORDS = td.train(td.words(file('../../resources/corpus.txt').read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in Trained_WORDS)

def known(words): 
    return set(w for w in words if w in Trained_WORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    return max(candidates, key=Trained_WORDS.get)

def correct_top(word, n):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    s = sorted(candidates, key=Trained_WORDS.get, reverse=True)
    return s[0], s[:n]

if __name__ == "__main__":
    print spell('hlo')
    
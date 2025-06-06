# -*- coding: utf-8 -*-
"""Abbreviations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PzkHIKG_cqzdzMSXKsUAx8ejFjWtPx82
"""

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd

class Abb():
    word = set()
    dic = dict()
    def __init__(self):
        df = pd.read_csv('Abbreviations.csv')
        for abb,nor in zip(df['Abbreviations'],df['Normalization']):
            ls = abb.split(',')
            for wor in ls:
                self.word.add(wor)
                self.dic[wor] = nor
    def rep(self,text):
        if text in self.word:
            return self.dic[text]
        else:
            return text
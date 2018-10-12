# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:21:09 2018

Sanity testing preprocessing

@author: anilnarayan
"""

import preprocessing as prep

df_all_train = pd.read_json('./data/train.json').sample(10)

all_ingredients_texts = df_all_train['ingredients'].apply(prep.space_separated_list_of_cleaned_ingredients)

with open('./data/preprocessing_results.txt', 'w', encoding='utf-8') as f:
    for i in df_all_train['ingredients'].index:
        print('{0} -  {1} ==> {2}'.format( df_all_train.loc[i]['cuisine'], prep.pretty_print_list(df_all_train.loc[i]['ingredients']), all_ingredients_texts.loc[i] ), file=f)


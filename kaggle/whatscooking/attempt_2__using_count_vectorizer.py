import numpy as np
import pandas as pd
import datetime
import persistence as ps
import preprocessing as prep
from sklearn.feature_extraction.text import CountVectorizer

df_all_train = pd.read_json('./data/train.json')

print(df_all_train.head())

#Change this value to zero if you don't want to load data and one-hot encode from training set
load_from_files = 1

print('Starting at ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

if load_from_files == 1:
    # Load from Files, else skip
    print('Loading traing data from file...')

    print('Prep df_all_train STARTED... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #for index in X.index:
    #    for ingredient in X.iloc[index]['ingredients']:
    #        #print('Setting value of ', ingredient, ' in row with index ', str(index))
    #        X.set_value(index, ingredient, 1)    
    y = df_all_train['cuisine']
    X = df_all_train['ingredients'].apply(prep.space_separated_list_of_cleaned_ingredients)
    
    vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english')
    X = vectorizer.fit_transform(X)
    print(type(X))
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))  
    print(vectorizer.get_stop_words())          
    print('Prep df_all_train ENDED... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    #pickling one-hot encoded training dataset
    print('START Saving X and y to file... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    ps.save_obj(X, "X")
    ps.save_obj(y, "y")
    
    print('DONE Saving X and y to file... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    
else:
    print('STARTED Loading from file',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    X = ps.load_obj('X')
    y = ps.load_obj('y')
    
    print('DONE Loading from file',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


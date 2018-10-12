import numpy as np
import pandas as pd
import datetime
import persistence as ps

df_all_train = pd.read_json('./data/train.json')

#df_all_train = df_all_train.head(1000)

def convert_array_to_lower(a):
    a = [x.lower() for x in a]
    return a

df_all_train['ingredients'] = [convert_array_to_lower(x) for x in df_all_train['ingredients']]

#Get unique ingredients across all recipes of all cuisines
unique_ingredients = set([])
for ingredients in df_all_train.iloc[:,2]:
    unique_ingredients = unique_ingredients | set(ingredients)
#print(unique_ingredients)

def one_hot_encode(df_input, unique_ingredients):
    X = pd.DataFrame(0, index=np.arange(len(df_input)),columns=list(unique_ingredients))
    X = pd.merge(df_input, X, left_index=True, right_index=True)

    for index in X.index:
        for ingredient in X.iloc[index]['ingredients']:
            #print('Setting value of ', ingredient, ' in row with index ', str(index))
            if ingredient in unique_ingredients:
                X.set_value(index, ingredient, 1)
    
    X = X.drop(['id', 'ingredients','cuisine'], axis=1)

    return X

#print(df_all_train)

#Change this value to zero if you don't want to load data and one-hot encode from training set
load_from_files = 1

print('Starting at ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

if load_from_files == 1:
    # Load from Files, else skip
    print('Loading traing data from file...')
    #pd.DataFrame(0, index=np.arange(len(data)), columns=fealeft_index=trueture_list)
    #X = pd.DataFrame(0, index=np.arange(len(df_all_train)),columns=list(unique_ingredients))
    #X = pd.merge(df, X, left_index=True, right_index=True)

    print('Prep and One-hot encoding of df_all_train STARTED... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #for index in X.index:
    #    for ingredient in X.iloc[index]['ingredients']:
    #        #print('Setting value of ', ingredient, ' in row with index ', str(index))
    #        X.set_value(index, ingredient, 1)    
    y = df_all_train['cuisine']
    X = one_hot_encode(df_all_train, unique_ingredients)
    print('Prep and One-hot encoding of df_all_train ENDED... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    
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
    
    
#print(X.shape)    
#print(y.shape)
#Look at the training data and remove unnecessary columns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(X.columns)

#Split the datasets into a 90% train and 10% test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Build the pipeline we'll use for scoring different classifiers
from sklearn.metrics import accuracy_score

def fit_and_predict_using_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier = classifier.fit(X_train.values, y_train.values)
    prediction_train = classifier.predict(X_train)
    prediction_test = classifier.predict(X_test)
    
    result = {}
    result['train_accuracy'] = accuracy_score(y_train, prediction_train)
    result['test_accuracy'] = accuracy_score(y_test, prediction_test)
    
    return result
    
from sklearn.linear_model import LogisticRegression
classifier_logistic_regression = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

result = fit_and_predict_using_classifier(classifier_logistic_regression, X_train, X_test, y_train, y_test)
#{'train_accuracy': 0.889931835959325, 'test_accuracy': 0.7830568124685772}
print(result)
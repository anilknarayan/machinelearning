import numpy as np
import pandas as pd
import datetime
import persistence as ps
import preprocessing as prep
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df_all_train = pd.read_json('./data/train.json')

print(df_all_train.head())

#Change this value to zero if you don't want to load data and one-hot encode from training set
load_from_files = 1

print('Starting at ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

if load_from_files == 0:
    # Load from Files, else skip
    print('Loading traing data from file...')

    print('Prep df_all_train STARTED... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #for index in X.index:
    #    for ingredient in X.iloc[index]['ingredients']:
    #        #print('Setting value of ', ingredient, ' in row with index ', str(index))
    #        X.set_value(index, ingredient, 1)    
    y = df_all_train['cuisine']
    X = df_all_train['ingredients'].apply(prep.space_separated_list_of_cleaned_ingredients)
    
    # START Using count vectorizer
    #vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english')
    #vectorizer_output = vectorizer.fit_transform(X)
    #X = pd.DataFrame(data=vectorizer_output.toarray(), columns=vectorizer.get_feature_names())
    #pd.DataFrame(data=doc_array, columns=count_vector.get_feature_names())
    # END Using count vectorizer
    
    # START Using TfIdf vectorizer
    vectorizer = TfidfVectorizer(binary=True)
    vectorizer_output = vectorizer.fit_transform(X)
    X = pd.DataFrame(data=vectorizer_output.toarray(), columns=vectorizer.get_feature_names())
    # END Using TfIdf vectorizer
    
    
    print(type(X.head()))
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

print(X.shape)
print(y.shape)

#Split the datasets into a 90% train and 10% test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#print(X_train.columns)

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

#print('STARTED Logistic Regression solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))     
#from sklearn.linear_model import LogisticRegression
#classifier_logistic_regression = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
#
#result = fit_and_predict_using_classifier(classifier_logistic_regression, X_train, X_test, y_train, y_test)
##Count Vectorizer : {'train_accuracy': 0.8580847022013632, 'test_accuracy': 0.7870789341377576}
#print('DONE Logistic Regression solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) 
#print(result)

#print('STARTED GaussianNB solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#from sklearn.naive_bayes import GaussianNB
#classifier_GaussianNB = GaussianNB()
#result = fit_and_predict_using_classifier(classifier_GaussianNB, X_train, X_test, y_train, y_test)
#{'train_accuracy': 0.3022684098781987, 'test_accuracy': 0.2420814479638009}
#only took one second to train and predict
#print('DONE GaussianNB solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#print(result)


#print('STARTED AdaBoost solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#classifier_adaBoost = AdaBoostClassifier(random_state=1)
#result = fit_and_predict_using_classifier(classifier_adaBoost, X_train, X_test, y_train, y_test)
##{'train_accuracy': 0.5583864118895966, 'test_accuracy': 0.5449974861739567}
##only took one second to train and predict
#print('DONE AdaBoost solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#print(result)

#print('STARTED Decision Tree solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#from sklearn.tree import DecisionTreeClassifier
#classifier_DecisionTree = DecisionTreeClassifier(random_state=1, criterion='entropy')
#result = fit_and_predict_using_classifier(classifier_DecisionTree, X_train, X_test, y_train, y_test)
##Count Vectorizer : {'train_accuracy': 0.5583864118895966, 'test_accuracy': 0.5449974861739567}
##only took one second to train and predict
#print('DONE Decision Tree solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
#print(result)

print('STARTED SVC... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))    
from sklearn.svm import SVC
classifier_SVC = SVC(C=50, # penalty parameter, setting it to a larger value 
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value, not tuned yet
	 			 gamma=1.4, # kernel coefficient, not tuned yet
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	 			 probability=False, # no need to enable probability estimates
	 			 cache_size=200, # 200 MB cache size
	 			 class_weight=None, # all classes are treated equally 
	 			 verbose=False, # print the logs 
	 			 max_iter=-1, # no limit, let it run
	 			 decision_function_shape=None, # will use one vs rest explicitly 
	 			 random_state=1)
result = fit_and_predict_using_classifier(classifier_SVC, X_train, X_test, y_train, y_test)
#Count Vectorizer : {'train_accuracy': 0.5583864118895966, 'test_accuracy': 0.5449974861739567}
#only took one second to train and predict
print('DONE SVC solution... ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))   
print(result)
#Count Vectorizer : {'train_accuracy': 0.5562073974745781, 'test_accuracy': 0.5399698340874811}
#TFIdf Vectorizer with optimized hyper-parameters - {'train_accuracy': 0.9996088948485864, 'test_accuracy': 0.8197586726998491}

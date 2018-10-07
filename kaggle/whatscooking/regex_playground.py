import re
import pandas as pd
import numpy as np

'''Remove strings in parenthesis'''
def remove_strings_in_parentheses(s):
    regex = r'\(([^()]*)\)'

    return re.sub(regex, '', s)

def remove_percentages(s):
    regex = r'[0-9]*%'
    return re.sub(regex, '', s)

def remove_trademarked_words(s):
    trademark_symbol = u"\u2122"
    s = s.replace(trademark_symbol,'(T)')
    regex = r'\b.*\(T\)'
    return re.sub(regex, '', s)

def remove_non_numeric_and_non_alphabetic_characters(s):
    regex = '[^\w^\s]'
    return re.sub(regex, '', s)

def space_separated_list_of_cleaned_ingredients(ingredients):
    cleaned_ingredients = []
    for ingredient in ingredients:
        cleaned_ingredient = remove_strings_in_parentheses(ingredient)
        cleaned_ingredient = remove_percentages(cleaned_ingredient)
        cleaned_ingredient = remove_trademarked_words(cleaned_ingredient)
        cleaned_ingredient = remove_non_numeric_and_non_alphabetic_characters(cleaned_ingredient)
        cleaned_ingredient = re.sub(r'[\s]+', ' ', cleaned_ingredient)
        cleaned_ingredient = cleaned_ingredient.lower()
        for sub_ingredient in cleaned_ingredient.split(' '):
            if len(sub_ingredient) > 2:
                cleaned_ingredients.append(sub_ingredient)
    return ' '.join(cleaned_ingredients)

def pretty_print_list(l):
    return '[ {} ]'.format( ', '.join(l) )

#print(pretty_print_list(['1','2','3','4']))

df = pd.read_json('./data/train.json')
#df = df.head(20)

for ingredients_array in df.ingredients:
    #ingredient = ingredient.replace('purÃ©e','puree')
    modified_ingredients_string = space_separated_list_of_cleaned_ingredients(ingredients_array)
    print('{0} --> {1}'.format(pretty_print_list(ingredients_array), modified_ingredients_string), file=open('./data/ingredient_preprocessing.csv', 'a'))
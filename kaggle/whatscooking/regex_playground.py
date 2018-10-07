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
        for sub_ingredient in cleaned_ingredient.split(' '):
            if len(sub_ingredient) > 2:
                cleaned_ingredients.append(sub_ingredient)
    return ' '.join(cleaned_ingredients)

#test_string = 'What if (and this is a big if) you had as many as 4 developers (maybe even 5 developers) reporting to you.'
test_string = 'I stopped using 2% reduced fat milk. Now I use 1% reduced fat milk. That has mademe 99% healthier.'


print(test_string2)

#print(remove_percentages(test_string))
#sheepâ€™s milk cheese
#Zatarainâ€™s Jambalaya Mix
#Breakstoneâ€™s Sour Cream

trademark_symbol = u"\u2122"

df = pd.read_json('./data/train.json')
df = df.head(20)

for ingredients_array in df.ingredients:
    #ingredient = ingredient.replace('purÃ©e','puree')
    modified_ingredients_string = space_separated_list_of_cleaned_ingredients(ingredients_array)
    print(*ingredients_array + ' -->' + modified_ingredients_string)

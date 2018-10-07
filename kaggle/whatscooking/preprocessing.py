import re
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:26:16 2018

All helper functions to convert an array of ingredients into a space separated 
list

@author: anilnarayan
"""

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
    regex = '[^\w^\s^-]'
    intermediate = re.sub(regex, '', s)
    return intermediate.replace('-', ' ')

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


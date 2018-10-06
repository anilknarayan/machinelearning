import numpy as np
import pandas as pd

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

unique_ingredients_list = list(unique_ingredients)
unique_ingredients_list.sort()

df = pd.DataFrame(unique_ingredients_list)
df.to_csv('./data/unique_ingredients.csv')

# After visual analysis, I see that we can perform the following clean-up on the ingredients -
#Remove all text in-between brackets
# Remove % and any number preceding them
# Remove all characters not a-z or A-Z
# Remove all words 2 characters or smaller
# remove stop words
# remove words not in lexicon - this should remove most brand names
# run these and visually inspect the list again to see if it looks better

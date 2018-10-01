# START Functions to persist and retrieve data to save time
import pickle    
def save_obj(obj, name ):
    with open(r'C:\tempjunk\intermediate\' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(r'C:\tempjunk\intermediate\' + name + '.pkl', 'rb') as f:
         return pickle.load(f)
# END Functions to persist and retrieve data to save time

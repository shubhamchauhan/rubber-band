import pickle
import numpy

def processed_data():
    pkl_file = open("pos_data.pkl","rb")
    lst = pickle.load(pkl_file)
    for ele in lst:
        try:
            for x in ele:
                ele[x][0]
                ele[x][1]
                ele[x][2]
        except:
            print "not a valid file"
            break

    print "Done Checking"


pkl_file = open("pos_trained.pkl","rb")
lst = pickle.load(pkl_file)
print set(lst[1].get_columns())-set(lst[2].get_columns())
    
"""

processed_data()
"""

import pickle
import pandas as pd
import sys

#Get data path
data_path = sys.argv[1]
#Load csv
new_data = pd.read_csv(data_path)
print(new_data)
#Load model
pickled_rf = pickle.load(open('rf.pkl', 'rb'))


print("Predicted class label(s): ")
print(pickled_rf.predict(new_data))

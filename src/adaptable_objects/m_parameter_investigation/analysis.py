import pickle
import os

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

for file in os.listdir(data_dir):
    file = file.decode("utf-8")

    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    print(data)
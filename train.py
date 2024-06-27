from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from pprint import pprint

data_file = pickle.load(open("./data.pickle", "rb"))

data = np.asarray(data_file["data"])
labels = np.asarray(data_file["labels"])

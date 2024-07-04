from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load the data and labels from the pickle file
data_file = pickle.load(open("./data.pickle", "rb"))

data = np.asarray(data_file["data"])  # Convert data to a NumPy array
labels = np.asarray(data_file["labels"])  # Convert labels to a NumPy array

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, shuffle=True, test_size=0.2)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print(f"{score * 100}% classified correctly")

# Save the trained model to a pickle file
with open("model.pickle", "wb") as f:
    pickle.dump({"model": model}, f)

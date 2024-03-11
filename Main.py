import numpy as np
import pandas as pd
import sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


with open("plik_tekst.txt", "r") as plik:   # wczytanie i dostosowanie danych treningowych
    train = plik.readlines()

train_data = []
for line in train:
    values = line.strip().strip("{").split(',')
    values[4] = values[4].strip('}')
    if len(values) > 5:
        values.pop()
    line_float = [float(x) for x in values]
    train_data.append(line_float)

with open("test-head7-three.txt", "r") as plik:   # wczytanie i dostosowanie danych testowych
    test = plik.readlines()

test_data = []
for line in test:
    vals = line.strip().split(",")
    test_data.append(vals)

# zamiana nazw poprawnych danych na numery

str_dict = {}
counter = 0

for row in test_data:
    tmp = row[-1]
    if tmp not in str_dict:
        str_dict[tmp] = counter
        counter += 1

for row in test_data:
    row[-1] = float(str_dict[row[-1]])

for i in test_data:
    for j in range(0, len(i)-1):
        i[j] = float(i[j])


train_data = np.array(train_data)
test_data = np.array(test_data)

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]


knn_classifier = KNeighborsClassifier(n_neighbors=2)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

print("Accuracy: {:.10f}".format(accuracy_score(y_test, y_pred)))


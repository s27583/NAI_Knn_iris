import math

from sklearn.metrics import accuracy_score


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)



def knn(train_data, test_data, k):
    pred_list = []
    for test_point in test_data:
        dist_list = []
        for train_point in train_data:
            distance = euclidean_distance(test_point[:-1], train_point[:-1])  # obliczanie odleglosci poza labelami
            dist_list.append((train_point, distance))

        dist_list.sort(key=lambda x: x[1])                                    # sortowanie wg ogleglosci
        neighbors = dist_list[:k]                                             # wybieranie pierwszych k elementow
        neighbor_labels = [neighbor[0][-1] for neighbor in neighbors]         # labele najblizszych sasiadow
        prediction = max(set(neighbor_labels), key=neighbor_labels.count)     # maksymalna wartosc
        pred_list.append(prediction)

    return pred_list



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


pred = knn(train_data, test_data, 3)

true_labels = [float(row[-1]) for row in test_data]  # rzeczywiste etykiety
accuracy = accuracy_score(true_labels, pred)         # obliczenie dokladnosci
print("dokladnosc z danymi testowymi z pliku testowego:", accuracy)


# Dane testowe


train_len = len(train_data)
test_from_train = int(0.1 * train_len)

for i in range(train_len - test_from_train, train_len - 1):
    test_data[i - train_len] = train_data[i]


pred = knn(train_data, test_data, 3)

true_labels = [float(row[-1]) for row in test_data]  # rzeczywiste etykiety
accuracy = accuracy_score(true_labels, pred)         # obliczenie dokladnosci
print("dokladnosc z danymi testowymi z pliku treningowego:", accuracy)



# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# train_data = np.array(train_data)
#
# X = train_data[:, :-1]
# y = train_data[:, -1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# knn_classifier = KNeighborsClassifier(n_neighbors=3)
# knn_classifier.fit(X_train, y_train)
#
# y_pred = knn_classifier.predict(X_test)
# print("dokladnosc:", accuracy_score(y_test, y_pred))
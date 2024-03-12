import math


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def k_nearest_neighbors(train_data, test_data, k):
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



pred = k_nearest_neighbors(train_data, test_data, 3)

# Obliczenie dokładności
correct_labels = [test_point[-1] for test_point in test_data]
accuracy = sum(1 for pred, label in zip(pred, correct_labels) if pred == label) / len(correct_labels)

print("Accuracy:", accuracy)

# train_data = np.array(train_data)
# test_data = np.array(test_data)
#
# X_train, y_train = train_data[:, :-1], train_data[:, -1]
# X_test, y_test = test_data[:, :-1], test_data[:, -1]
#
#
# knn_classifier = KNeighborsClassifier(n_neighbors=3)
#
# knn_classifier.fit(X_train, y_train)
#
# y_pred = knn_classifier.predict(X_test)
#
# acc = accuracy_score(y_test, y_pred)


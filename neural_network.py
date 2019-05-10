import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import itertools as it

rows_to_read = 5000
input_file = 'converted_seq_test.csv'
# 'kmer_positive_training.csv'
# column headings
# dictionary of kmers
all_bases = 'ACGT'
keywords = it.product(all_bases, repeat=5)
# print(type(keywords))
all_combinations = list(keywords)
headers = []
headers2 = []
for tup in all_combinations:
    headers.append(''.join(tup))
for tup in all_combinations:
    headers.append(''.join(tup))

for n in range(0, 18):
    headers.append("Type" + str(int(n)) + str(1))

headers.append("Label1")

# read data
print(str(headers))
print(str(headers2))

kmer_neg_data = pd.read_csv(input_file, error_bad_lines=False,
                            names=headers)
kmer_pos_data = pd.read_csv(input_file, error_bad_lines=False,
                            names=headers)
print(len(kmer_pos_data.columns))

kmer_full_data = pd.concat([kmer_neg_data, kmer_pos_data], join='outer', sort=False)

print((kmer_full_data.head()))
features = kmer_full_data.iloc[:, 0: 2048]
type1 = kmer_full_data.iloc[:, 2048: 2057]
type2 = kmer_full_data.iloc[:, 2057: 2066]
# type1.applymap(str)
#
# type2.applymap(str)
# print((type1.head()))
# print((type2.head()))

x = type1.to_string(header=False,
                    index=False,
                    ).split('\n')
type1 = [''.join(ele.split()) for ele in x]
print(type1)

x = type2.to_string(header=False,
                    index=False,
                    ).split('\n')
type2 = [''.join(ele.split()) for ele in x]
print(type2)

type_list1 = []
type_list2 = []
# print(type1)

for n in range(0, len(type1)):
    type_list1.append(int(type1[n], 2))
    type_list2.append(int(type2[n], 2))

print(len(type_list2))
print(len(type_list1))

print(len(features))

features['Type1'] = type_list1
features['Type2'] = type_list2

print(features.head())

# type1.stack().groupby(level=0).apply(''.join)
# type2.stack().groupby(level=0).apply(''.join)
#
# for n in range(0, len(type1) - 1):
#     s1 = pd.Series(type1.iloc[n, :])
#     s2 = pd.Series(type2.iloc[n, :])
#     print(s1, s2)
#     type_list1.append(s1, 2)
#     type_list2.append(s2, 2)

label = kmer_full_data.loc[:, 'Label1']

x_train, x_test, y_train, y_test = train_test_split(features, label)
np.set_printoptions(precision=3)

n_iterations = 500
hidden_layers = (50, 50, 50)

scaler = StandardScaler()
scaler.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=n_iterations)
mlp.fit(x_train, y_train)

MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=hidden_layers, learning_rate='constant',
              learning_rate_init=0.001, max_iter=n_iterations, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=None,
              shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,
              verbose=False, warm_start=False)

predictions = mlp.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
# clf.fit(x_train, y_train)

# print("Accuracy:", clf.score(x_test, y_test))

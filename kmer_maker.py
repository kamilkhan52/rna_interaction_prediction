import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from time import time
import re
import pickle
import itertools as it
import csv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# dictionary of kmers
all_bases = 'ACGT'
keywords = it.product(all_bases, repeat=5)
# print(type(keywords))
all_combinations = list(keywords)
', '.join(keywords)
dictOfKmers = {i: 0 for i in all_combinations}


# print(dictOfKmers)


def convert_seq_to_kmer():
    seq_number = 1
    full_row = []
    f = open('converted_seq_test.csv', 'w')
    row = dictOfKmers.copy()
    for j in range(0, len(seq_input_data.index)):
        for i in range(0, 2):
            try:
                for w in range(0, len(seq_input_data[i][j]) - 4):
                    current_string = seq_input_data[i][j][w:w + 5]
                    # print(current_string)
                    # row[tuple(current_string)] = row[tuple(current_string)] + 1  # frequency kmers
                    row[tuple(current_string)] = 1  # yes/no kmer

                # print("end of one seq")
                if seq_number < 2:  # if we are on the same row
                    for value in row.values():
                        full_row.append(value)
                    seq_number += 1
                    # print(seq_number)
                else:  # if the this is the last sequence on the row
                    for value in row.values():
                        full_row.append(value)
                    for t in range(1, 19):
                        full_row.append(0)
                    full_row.append(seq_input_data[len(seq_input_data.columns) - 1][j])
                    w = csv.writer(f)
                    w.writerow(full_row)
                    # print("written a row of length: " + str(len(full_row)))
                    full_row.clear()
                    seq_number = 1
                    row.clear()
                    row = dictOfKmers.copy()
            except:  # bad line
                pass


#
seq_input_data = pd.read_csv('seq_test.csv', header=None)

kmer_neg_data = pd.read_csv('seq_negative_training.csv', nrows=15000, header=None)
kmer_neg_data[4] = 'N'
print(kmer_neg_data.head())

kmer_neg_data = kmer_neg_data.dropna()

kmer_pos_data = pd.read_csv('seq_positive_training.csv', nrows=15000, header=None)
kmer_pos_data[4] = 'P'
print(kmer_pos_data.head())

kmer_pos_data = kmer_pos_data.dropna()

seq_input_data = pd.concat([kmer_neg_data, kmer_pos_data], sort=False, ignore_index=False)
seq_input_data = seq_input_data.dropna()
seq_input_data = seq_input_data.reset_index(drop=True)
print(seq_input_data.head())

print(seq_input_data[len(seq_input_data.columns) - 1][3])

convert_seq_to_kmer()
#

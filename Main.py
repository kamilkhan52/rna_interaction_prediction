import numpy as np
import pandas as pd
from scipy.spatial import distance
import math as math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn import decomposition
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def setup_random_search():
    global random_grid
    # set up hyper parameter search
    # The range and step-size for the number of trees
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Feature selection algorithms to explore
    max_features = ['auto', 'sqrt']
    # The range and step-size for the max number of levels in a tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples requirement for node-splitting
    min_samples_split = [2, 5, 10]
    # Minimum number of samples requirement for declaring leaf-node
    min_samples_leaf = [1, 2, 4]
    # Bootstrap enable/disable
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)


def run_random_search():
    global random_grid
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iterations, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    start = time()
    rf_random.fit(x_train, y_train)

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iterations))
    report(rf_random.cv_results_)
    print(rf_random.best_score_)


rows_to_read = 10000
# read data
# kmer_neg_data = pd.read_csv('kmer_negative_training.csv', nrows=rows_to_read, header=None, error_bad_lines=False)
# kmer_pos_data = pd.read_csv('kmer_positive_training.csv', nrows=rows_to_read, header=None, error_bad_lines=False)
# print(len(kmer_pos_data.columns))
#
# kmer_full_data = pd.concat([kmer_pos_data, kmer_neg_data], ignore_index=True, join='outer')
kmer_full_data = pd.read_csv('converted_seq_test.csv', header=None,
                             error_bad_lines=False)  # reading from kmer_maker

features = kmer_full_data.iloc[:, 0: 2066]
label = kmer_full_data.iloc[:, 2066]
print(label)
x_train, x_test, y_train, y_test = train_test_split(features, label)
np.set_printoptions(precision=3)

n_iterations = 10
n_comp = 200

clf = RandomForestClassifier(n_estimators=10000)

# Train the model using the training sets
clf.fit(x_train, y_train)

print("Accuracy:", clf.score(x_test, y_test))

# setup_random_search()
# run_random_search()

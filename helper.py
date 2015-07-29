__author__ = 'diego'

import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pickle

default_feature_names = np.array(["feat_%d" % i for i in range(1,94)])


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')

    with open(name+".pickle", 'wb') as f:
        pickle.dump(clf, f, -1)
    print("Wrote submission to file {}.".format(name))

def select_features(x, y, fn=default_feature_names):
    total_features = len(fn)
    forest = ExtraTreesClassifier(n_estimators=10,
                                 random_state=0)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    x_indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(fn)):
        print("%d. feature %s (%f)" % (f + 1, fn[x_indices[f]], importances[x_indices[f]]))
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(total_features), importances[x_indices],
            color="r", yerr=std[x_indices], align="center")
    plt.xticks(range(total_features), fn[x_indices])
    plt.xlim([-1, total_features])
    plt.show()
    return x_indices
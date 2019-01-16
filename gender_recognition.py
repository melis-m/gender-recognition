#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree
import sklearn.feature_selection
import sklearn.linear_model
import sys


def similarity(a, b):
    x = 0
    length = len(a)
    for i in range(length):
        x += a[i] == b[i]
    return x / length


def test(classifier, X_test, y_test):
    res = classifier.predict(X_test)
    sim = similarity(res, y_test)
    return sim


def train(classifier, dataframe):
    features = dataframe.values[:, :-1]
    class_labels = [x[0] for x in dataframe.values[:, -1:]]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            features, class_labels, test_size=0.1,
    )

    classifier.fit(X_train, y_train)
    return X_test, y_test, X_train, y_train


def create_svm():
    return sklearn.svm.SVC(gamma=0.22, C=8)


def create_decision_tree():
    return sklearn.tree.DecisionTreeClassifier()


def create_random_forest(nb_trees=10):
    return sklearn.ensemble.RandomForestClassifier(n_estimators=nb_trees)


def pick_classifier(arg):
    if arg == 'rf':
        return create_random_forest
    elif arg == 'dt':
        return create_decision_tree
    elif arg == 'svm':
        return create_svm


def main():
    dataframe = pd.read_csv('voice.csv')

    sims = []
    trains = []
    create_classifier = pick_classifier(sys.argv[1]) if len(sys.argv) >= 2 \
        else create_random_forest
    for _ in range(10):
        classifier = create_classifier()
        X_test, y_test, X_train, y_train = train(classifier, dataframe)
        t = test(classifier, X_train, y_train)
        print('train accuracy: {}'.format(t))
        trains.append(t)
        print('mean train accuracy: {}'.format(np.mean(trains)))
        s = test(classifier, X_test, y_test)
        print('test accuracy: {}'.format(s))
        sims.append(s)
        print('mean test accuracy: {}\n'.format(np.mean(sims)))


if __name__ == '__main__':
    main()

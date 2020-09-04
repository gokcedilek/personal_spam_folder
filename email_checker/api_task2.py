from sklearn.feature_extraction.text import HashingVectorizer
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import csv
import os

# define constants
CV_FOLDS = 5
stats = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']


def retrieve_emails(cursor, tablename):
    try:
        cursor.execute(f"select sender, subject, label from {tablename}")
        data = cursor.fetchall()
        cursor.execute(f"select count(*) from {tablename}")
        count = cursor.fetchone()[0]
        return data, count
    except Exception as e:
        print(f'Exception: {e}\n')


def hash_encode(training, test):
    # extract fields in test & training data
    train_sen = [t[0] for t in training]
    test_sen = [t[0] for t in test]
    train_sub = [t[1] for t in training]
    test_sub = [t[1] for t in test]
    train_label = [t[2] for t in training]
    test_label = [t[2] for t in test]

    # define encoders
    sen_vectorizer = HashingVectorizer(
        n_features=20, norm=None, alternate_sign=False)
    sub_vectorizer = HashingVectorizer(
        n_features=20, norm=None, alternate_sign=False)

    # encode training data
    train_sen_encoded = sen_vectorizer.fit_transform(train_sen)
    train_sub_encoded = sub_vectorizer.fit_transform(train_sub)

    # encode test data
    test_sen_encoded = sen_vectorizer.transform(test_sen)
    test_sub_encoded = sub_vectorizer.transform(test_sub)

    # create training & test feature matrices
    train_features = sp.hstack(
        [train_sen_encoded, train_sub_encoded], format='csr')
    test_features = sp.hstack(
        [test_sen_encoded, test_sub_encoded], format='csr')
    return train_features, train_label, test_features, test_label


def num_folds(train_label):
    # determine the number of folds for cross-validation of training set - there should be at least one member from each class in each fold!
    count_0 = train_label.count(0)
    count_1 = train_label.count(1)
    min_occurrence = min(count_0, count_1)
    return CV_FOLDS if CV_FOLDS < min_occurrence else min_occurrence


def fit_clf(train_features, train_label):
    # define hyperparameters for the Multinomial NB classifier
    params = {
        'alpha': [0.01, 0.1, 1, 2],
        'class_prior': [None],
        'fit_prior': [True, False]
    }
    # calculate number of folds for cross-validation
    cv = num_folds(train_label)
    # tune hyperparameters via grid-search cross-validation
    clf = GridSearchCV(estimator=MultinomialNB(),
                       param_grid=params, cv=cv, scoring=stats, refit='accuracy')
    # train the tuned model
    clf.fit(train_features, train_label)
    return clf


def get_cv_stats(clf):
    # create a dictionary of stats for the training data
    # NOTE: clf.cv_results_ is a dictionary whose values are arrays. first, extract the keys we want (mean_test_<metric>)
    # then, extract the array item from the value array, to only choose the item that belongs to the (combination of) hyperparams that led to the best accuracy (clf.best_index_), because those hyperparams are used for training & testing
    training_dict = {metric: clf.cv_results_[
        f'mean_test_{metric}'][clf.best_index_] for metric in stats}
    return training_dict


def get_pred_stats(clf, test_features, test_label):
    # make predictions for the testing data
    pred = clf.predict(test_features)
    # compute stats for the testing data
    accuracy = metrics.accuracy_score(test_label, pred)
    balanced_accuracy = metrics.balanced_accuracy_score(test_label, pred)
    precision = metrics.precision_score(test_label, pred)
    recall = metrics.recall_score(test_label, pred)
    f1 = metrics.f1_score(test_label, pred)
    # create a dictionary of stats for the testing data
    test_metrics = [accuracy, balanced_accuracy, precision, recall, f1]
    test_dict = {stats[i]: test_metrics[i] for i, metric in enumerate(stats)}
    return test_dict


def save_stats(filename, openmode, stats_dict):
    # save training / test stats to csv
    header = False
    if not os.path.isfile(filename):
        # if first time writing to the file, add headers
        header = True
    with open(filename, openmode) as stats_csv:
        headers = list(stats_dict.keys())
        writer = csv.DictWriter(stats_csv, fieldnames=headers)
        if header:
            writer.writeheader()
        writer.writerow(stats_dict)

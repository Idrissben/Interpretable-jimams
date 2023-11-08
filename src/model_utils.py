import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle

def fit_dummy_classifier(X_train, y_train):
    dummy_classifier = DummyClassifier(strategy='most_frequent')
    dummy_classifier.fit(X_train, y_train)
    return dummy_classifier

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, balanced_accuracy, roc_auc, f1

def grid_search(estimator, X_train, y_train, param_grid, cv, scoring):
    grid_search = GridSearchCV(estimator, param_grid, cv, scoring)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    return best_estimator

def save_model(model, path):
    pickle.dump(model, open(path, "wb"))

def load_saved_model(path):
    model = pickle.load(open(path, "rb"))
    return model
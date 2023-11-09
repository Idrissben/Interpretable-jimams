import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from pygam import LogisticGAM
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle
from typing import Tuple

def fit_dummy_classifier(X_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    """
    Fit a dummy classifier with the 'most_frequent' strategy.

    Args:
        X_train: Training data (features).
        y_train: Training labels.

    Returns:
        DummyClassifier: The fitted dummy classifier.
    """
    dummy_classifier = DummyClassifier(strategy='most_frequent')
    dummy_classifier.fit(X_train, y_train)
    return dummy_classifier

def fit_logistic_gam(X_train: np.ndarray, y_train: np.ndarray) -> LogisticGAM:
    """
    Fit a Logistic GAM (Generalized Additive Model) classifier.

    Args:
        X_train: Training data (features).
        y_train: Training labels.

    Returns:
        LogisticGAM: The fitted Logistic GAM model.
    """
    gam = LogisticGAM()
    gam.fit(X_train, y_train)
    return gam

def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate a classification model using various metrics.

    Args:
        model: The trained classification model.
        X: Input data for prediction.
        y: True labels.

    Returns:
        Tuple[float, float, float, float]: A tuple containing accuracy, balanced accuracy, ROC AUC, and F1 score.
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, balanced_accuracy, roc_auc, f1

def grid_search(estimator, X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, cv, scoring: str):
    """
    Perform a grid search for hyperparameter tuning.

    Args:
        estimator: The classifier or regression model to tune.
        X_train: Training data (features).
        y_train: Training labels.
        param_grid: Dictionary of hyperparameters to search.
        cv: Cross-validation strategy.
        scoring: Scoring metric for evaluation.

    Returns:
        object: The best estimator found by grid search.
    """
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    return best_estimator

def save_model(model, path: str):
    """
    Save a trained model to a file using pickle.

    Args:
        model: The trained model to save.
        path: The file path to save the model.
    """
    pickle.dump(model, open(path, "wb"))

def load_saved_model(path: str):
    """
    Load a saved model from a file using pickle.

    Args:
        path: The file path from which to load the model.

    Returns:
        object: The loaded model.
    """
    model = pickle.load(open(path, "rb"))
    return model
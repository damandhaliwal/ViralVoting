# import libraries for Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error
from data_clean import clean_data
from utils import get_project_paths, select_features, prepare_features
import numpy as np
import pandas as pd
from typing import Literal, Optional

def gradient_boosting_classifier(
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_leaf: int = 200,
    subsample: float = 1.0,
    feature_strategy: Literal["minimal", "core", "all"] = "minimal",
    random_state: int = 42,
) -> dict:
    """
    Fit a Gradient Boosting classifier.
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting stages
    learning_rate : float
        Shrinks contribution of each tree
    max_depth : int
        Maximum depth of individual trees
    min_samples_leaf : int
        Minimum samples at leaf nodes
    subsample : float
        Fraction of samples for fitting each tree (stochastic gradient boosting)
    feature_strategy : {"minimal", "core", "all"}
        Feature selection strategy
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Model, metrics (train/test accuracy and logloss), feature names
    """
    
    data = clean_data().copy()
    y = data["voted"].astype(int).values

    feature_cols = select_features(data, feature_strategy)
    X, feature_names = prepare_features(data, feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y, 
                                                        test_size=0.2, 
                                                        random_state=random_state, 
                                                        stratify=y)

    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    gb.fit(X_train, y_train)

    y_test_pred = gb.predict(X_test)
    y_test_proba = gb.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_logloss = log_loss(y_test, y_test_proba)

    y_train_pred = gb.predict(X_train)
    y_train_proba = gb.predict_proba(X_train)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_logloss = log_loss(y_train, y_train_proba)

    print(
        f"Gradient Boosting fitted (strategy={feature_strategy}, "
        f"n_estimators={n_estimators}, learning_rate={learning_rate}, "
        f"n_features={len(feature_names)}, n_samples_train={X_train.shape[0]}). "
        f"Test accuracy={test_accuracy:.4f}, Test log-loss={test_logloss:.4f}"
    )
    return {
        "model": gb,
        "feature_names": feature_names,
        "test_accuracy": test_accuracy,
        "test_logloss": test_logloss,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
    }
# import libraries for bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error
from data_clean import clean_data
from utils import get_project_paths, select_features, prepare_features
import numpy as np
import pandas as pd
from typing import Literal, Optional

def bagging_classifier(
    n_estimators: int = 300,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 200,
    feature_strategy: Literal["minimal", "core", "all"] = "core",
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> dict:
    """
    Fit a Bagging classifier using decision trees as base estimators.
    
    Parameters
    ----------
    n_estimators : int
        Number of base estimators (trees) in the ensemble
    max_samples : float
        Fraction of samples to draw for each base estimator
    max_features : float  
        Fraction of features to draw for each base estimator
    max_depth : int or None
        Maximum depth of base decision trees
    min_samples_leaf : int
        Minimum samples at leaf for base trees
    feature_strategy : {"minimal", "core", "all"}
        Feature selection strategy
    random_state : int
        Random seed for reproducibility
    class_weight : str or None
        Class weighting for base estimator
        
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

    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    
    bagging = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )
    bagging.fit(X_train, y_train)
    
    # TODO 6: Compute test metrics
    y_hat = bagging.predict(X_test)
    y_test_proba = bagging.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_logloss = log_loss(y_test, y_test_proba)
    
    # TODO 7: Compute train metrics  
    y_train_pred = bagging.predict(X_train)
    y_train_proba = bagging.predict_proba(X_train)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_logloss = log_loss(y_train, y_train_proba)
    
    # TODO 8: Print results
    print(
        f"Bagging fitted (strategy={feature_strategy}, "
        f"n_estimators={n_estimators}, n_features={len(feature_names)}, "
        f"n_samples_train={X_train.shape[0]}). "
        f"Test accuracy={test_accuracy:.4f}, Test log-loss={test_logloss:.4f}"
    )
    
    # TODO 9: Return results dictionary
    return {
        "model": bagging,
        "feature_names": feature_names,
        "test_accuracy": test_accuracy,
        "test_logloss": test_logloss,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
    }
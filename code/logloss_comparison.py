import numpy as np
import pandas as pd
from typing import Literal, Optional
from bagging import bagging_classifier
from boosting import gradient_boosting_classifier
from utils import get_project_paths, select_features, prepare_features
from tree_forest import random_forest_classifier, classification_tree

def compare_classifiers():
    """
    Compare Bagging, Gradient Boosting, Random Forest, and Decision Tree classifiers
    using accuracy and log-loss metrics on the same train/test split.
    
    Returns
    -------
    pd.DataFrame
        DataFrame summarizing test accuracy and log-loss for each classifier
    """
    results = []

    # Bagging Classifier
    bagging_results = bagging_classifier()
    results.append({
        "Model": "Bagging",
        "Test Accuracy": bagging_results["test_accuracy"],
        "Test Log-Loss": bagging_results["test_logloss"]
    })

    # Gradient Boosting Classifier
    boosting_results = gradient_boosting_classifier()
    results.append({
        "Model": "Gradient Boosting",
        "Test Accuracy": boosting_results["test_accuracy"],
        "Test Log-Loss": boosting_results["test_logloss"]
    })

    # Random Forest Classifier
    rf_results = random_forest_classifier()
    results.append({
        "Model": "Random Forest",
        "Test Accuracy": rf_results["test_accuracy"],
        "Test Log-Loss": rf_results["test_logloss"]
    })

    # Decision Tree Classifier
    dt_results = classification_tree()
    results.append({
        "Model": "Decision Tree",
        "Test Accuracy": dt_results["test_accuracy"],
        "Test Log-Loss": dt_results["test_logloss"]
    })
    print(results)
    return pd.DataFrame(results)

compare_classifiers()
import os
import pandas as pd
import numpy as np
from typing import Literal, Optional

def get_project_paths():
    """
    Returns a dictionary with paths to important directories in the project.
    
    Returns:
        dict: Dictionary with the following keys:
            - parent_dir: Parent directory of the code folder
            - data_dir: Path to the Data directory
            - plots_dir: Path to the Output/Plots directory
            - tables_dir: Path to the Output/Tables directory
    """
    # Get the parent directory of the current file
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)
    
    # Define paths relative to the parent directory
    data_dir = os.path.join(parent_dir, 'Data/')
    plots_dir = os.path.join(parent_dir, 'Output', 'Plots/')
    tables_dir = os.path.join(parent_dir, 'Output', 'Tables/')
    
    return {
        'parent_dir': parent_dir,
        'data': data_dir,
        'plots': plots_dir,
        'tables': tables_dir
    }

MINIMAL_FEATURES = [
    "sex", "yob", "g2000", "g2002", "g2004", "p2000", "p2002",
    "treatment_civic duty", "treatment_hawthorne", "treatment_neighbors", "treatment_self",
]

CORE_EXTRA_FEATURES = [
    "median_age", "median_income", "percent_white", 
    "percent_black", "hsorhigher", "bach_orhigher"
]

FEATURE_BLACKLIST = {
    "zip", "plus4", "city", "CityName", "hh_id", 
    "tract", "block", "geography", "id", "id2", "cluster", "voted"
}

def select_features(
    data: pd.DataFrame, 
    strategy: Literal["minimal", "core", "all"] = "core"
) -> list[str]:
    """Select features based on strategy.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    strategy : {"minimal", "core", "all"}
        Feature selection strategy
        
    Returns
    -------
    list[str]
        List of column names to use
    """
    if strategy == "minimal":
        candidate = MINIMAL_FEATURES
    elif strategy == "core":
        candidate = MINIMAL_FEATURES + CORE_EXTRA_FEATURES
    else:  # "all"
        candidate = [c for c in data.columns if c not in FEATURE_BLACKLIST]
    
    # Keep only existing columns
    return [c for c in candidate if c in data.columns]


def prepare_features(
    data: pd.DataFrame, 
    feature_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode and handle missing values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with selected features
    feature_cols : list[str]
        Columns to use as features
        
    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix and final feature names
    """
    X_df = data[feature_cols].copy()
    
    # One-hot encode categorical columns
    obj_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X_df = pd.get_dummies(X_df, columns=obj_cols, drop_first=True)
    
    # Fill missing values efficiently (vectorized)
    X_df = X_df.fillna(X_df.median())
    
    return X_df.values, list(X_df.columns)


def subsample_data(
    X: np.ndarray, 
    y: np.ndarray, 
    sample_size: Optional[int],
    random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample data if sample_size specified.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    sample_size : int or None
        Number of samples to draw (None = use all)
    random_state : int
        Random seed
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Subsampled X and y
    """
    if sample_size is None or sample_size >= X.shape[0]:
        return X, y
    
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=sample_size, replace=False)
    return X[idx], y[idx]

def get_clean_variable_names():
    return {
        'const': 'Constant',
        'voted': 'Voted',
        'treatment\_civic duty': 'Civic Duty',
        'treatment\_hawthorne': 'Hawthorne',
        'treatment\_neighbors': 'Neighbors',
        'treatment\_self': 'Self',
        'treatment\_control': 'Control',
        'sex': 'Female',
        'yob': 'Year of Birth',
        'g2000': 'Voted General 2000',
        'g2002': 'Voted General 2002',
        'g2004': 'Voted General 2004',
        'p2000': 'Voted Primary 2000',
        'p2002': 'Voted Primary 2002',
        'p2004': 'Voted Primary 2004',
        'treatment_intensity': 'Treatment Intensity',
        'high_block_intensity': 'High Intensity Block',
        'cluster_size': 'Cluster Size',
        'hh_size': 'Household Size',
        'treatment\_intensity': 'Treatment Intensity'
    }

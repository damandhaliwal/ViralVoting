import pymc as pm
import pymc_bart as pmb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from data_clean import clean_data
from utils import get_project_paths, select_features, prepare_features
import numpy as np
from typing import Literal, Optional

def bart_classifier(
    n_trees: int = 50,  # BART typically uses fewer trees
    n_samples: int = 2000,  # MCMC samples
    n_tune: int = 1000,  # Burn-in samples
    feature_strategy: Literal["minimal", "core", "all"] = "minimal",
    random_state: int = 42,
) -> dict:
    """
    Fit a BART (Bayesian Additive Regression Trees) classifier.
    
    Parameters
    ----------
    n_trees : int
        Number of trees in the sum-of-trees model
    n_samples : int
        Number of posterior samples to draw
    n_tune : int
        Number of tuning/burn-in samples
    feature_strategy : {"minimal", "core", "all"}
        Feature selection strategy
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Model, posterior samples, metrics, feature names
    """
    data = clean_data().copy()
    y = data["voted"].astype(int).values
    
    feature_cols = select_features(data, feature_strategy)
    X, feature_names = prepare_features(data, feature_cols)
    
    # TODO 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y, 
                                                        test_size=0.2, 
                                                        random_state=random_state, 
                                                        stratify=y)
    
    with pm.Model() as bart_model:
        # Define BART prior
        bart = pmb.BART(
            name="bart",
            X=X_train,
            Y=y_train,
            m=n_trees,  # number of trees
        )
        
        # Sample from posterior
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_state,
            chains=2,  # Multiple chains for convergence checking
            progressbar=True,
        )

        pm.sample_posterior_predictive(
            trace,
            var_names=["bart"],
            # Need to pass test data somehow - check pymc-bart docs
            extend_inferencedata=True,
        )
    
    # TODO 6: Extract predictions from posterior
    # trace.posterior_predictive contains samples
    # You need to aggregate them (mean? median?)
    # y_test_proba_samples = ??? (shape: n_samples x n_test_observations)
    # y_test_proba = ??? (aggregate to single prediction per observation)
    
    # TODO 7: Convert probabilities to class predictions
    # y_test_pred = ???
    
    # TODO 8: Compute test metrics
    test_accuracy = # ???
    test_logloss = # ???
    
    # TODO 9: Do the same for training set (need to generate train predictions)
    # train_accuracy = ???
    # train_logloss = ???
    
    # TODO 10: Print results
    print(
        f"BART fitted (strategy={feature_strategy}, "
        f"n_trees={n_trees}, n_samples={n_samples}, "
        f"n_features={len(feature_names)}). "
        f"Test accuracy={test_accuracy:.4f}, Test log-loss={test_logloss:.4f}"
    )
    
    # TODO 11: Return results
    return {
        "model": bart_model,
        "trace": trace,
        "feature_names": feature_names,
        "test_accuracy": test_accuracy,
        "test_logloss": test_logloss,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
    }
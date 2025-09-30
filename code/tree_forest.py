from sklearn.tree import DecisionTreeRegressor as DTR, DecisionTreeClassifier as DTC, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, root_mean_squared_error
from data_clean import clean_data
from utils import get_project_paths, select_features, prepare_features, subsample_data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Literal, Optional
from sklearn.model_selection import train_test_split


def save_tree_plot(
    model,
    feature_names: list[str],
    title: str,
    filename: str,
    class_names: Optional[list[str]] = None,
    figsize: tuple[int, int] = (18, 10)
) -> str:
    """Save tree visualization plot.
    
    Parameters
    ----------
    model : sklearn tree model
        Fitted tree model
    feature_names : list[str]
        Feature names for plot
    title : str
        Plot title
    filename : str
        Output filename (without path)
    class_names : list[str], optional
        Class names for classification trees
    figsize : tuple[int, int]
        Figure size
        
    Returns
    -------
    str
        Full path to saved plot
    """
    paths = get_project_paths()
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        impurity=True,
        rounded=True,
        fontsize=8,
    )
    ax.set_title(title)
    
    plot_path = paths["plots"] + filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved plot to: {plot_path}")
    return plot_path


def regression_tree(
    max_depth: int = 4,
    min_samples_leaf: int = 100,
    random_state: int = 0,
    save_plot: bool = True,
) -> dict:
    """Fit and (optionally) plot a regression tree predicting `voted`.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    min_samples_leaf : int
        Minimum number of samples required at a leaf node.
    random_state : int
        Reproducibility seed.
    save_plot : bool
        If True, saves a PNG to Output/Plots/.

    Returns
    -------
    dict
        Fitted model, feature names, train MSE, and plot path.
    """
    data = clean_data().copy()
    y = data["voted"].astype(float).values

    # Build feature matrix
    X_df = data.drop(columns=["voted"])
    X_df = pd.get_dummies(X_df, drop_first=True)
    feature_names = list(X_df.columns)
    X = X_df.values

    X_train,X_test,y_train,y_test = train_test_split(
                                                      X,y,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y)

    # Fit regression tree
    tree = DTR(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X_train, y_train)

    # Evaluate
    y_hat = tree.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_hat)
    train_rmse = root_mean_squared_error(y_train, tree.predict(X_train))

    # Plot
    plot_path = None
    if save_plot:
        plot_path = save_tree_plot(
            tree,
            feature_names,
            "Regression Tree Predicting Probability of Voting",
            "regression_tree.png"
        )

    print(
        f"Regression tree fitted. max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}, test_rmse={test_rmse:.4f}, train_rmse={train_rmse:.4f}"
    )

    return {
        "model": tree,
        "feature_names": feature_names,
        "test_rmse": test_rmse,
        "train_rmse": train_rmse,
        "plot_path": plot_path,
    }


def classification_tree(
    max_depth: int = 4,
    min_samples_leaf: int = 300,
    criterion: str = "gini",
    class_weight: str = "balanced",
    random_state: int = 0,
    save_plot: bool = True,
    feature_strategy: Literal["minimal", "core", "all"] = "core",
    sample_for_fit: Optional[int] = None,
    sample_for_plot: int = 8000,
) -> dict:
    """Fit and (optionally) plot a classification tree for `voted`.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    criterion : {"gini", "entropy", "log_loss"}
        Split criterion.
    class_weight : str or dict
        Handle class imbalance.
    random_state : int
        Random seed.
    save_plot : bool
        Whether to save plot.
    feature_strategy : {"minimal", "core", "all"}
        Feature selection strategy.
    sample_for_fit : int or None
        Sample size for fitting (None = all).
    sample_for_plot : int
        Sample size for plotting clarity.

    Returns
    -------
    dict
        Model, feature names, metrics, and plot path.
    """
    data = clean_data().copy()
    y_full = data["voted"].astype(int).values

    # Select and prepare features
    feature_cols = select_features(data, feature_strategy)
    X_full, feature_names = prepare_features(data, feature_cols)
    
    X_train,X_test,y_train,y_test = train_test_split(
                                                X_full,y_full,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=y_full)

    # Fit classifier
    clf = DTC(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_hat = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_logloss = log_loss(y_test, y_proba)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    train_logloss = log_loss(y_train, clf.predict_proba(X_train)[:,1])

    # Plot
    plot_path = None
    if save_plot:
      plot_clf = clf
      plot_path = save_tree_plot(
            plot_clf,
            feature_names,
            "Classification Tree Predicting Voting (Yes/No)",
            "classification_tree.png",
            class_names=["No", "Yes"],
            figsize=(30,16)
        )

    print(
        f"Classification tree fitted (strategy={feature_strategy}, "
        f"n_features={len(feature_names)}, n_samples={X_train.shape[0]}). "
        f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
        f"accuracy={test_accuracy:.4f}, log-loss={test_logloss:.4f}"
    )

    return {
        "model": clf,
        "feature_names": feature_names,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_logloss": test_logloss,
        "plot_path": plot_path,
    }


def random_forest_classifier(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 300,
    max_features: str = "sqrt",
    feature_strategy: Literal["minimal", "core", "all"] = "minimal",
    sample_for_fit: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    random_state: int = 0,
    top_n: int = 20,
    save_plot: bool = True,
) -> dict:
    """Fit a Random Forest classifier and plot feature importance.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Max depth for each tree.
    min_samples_leaf : int
        Min samples at a leaf.
    max_features : {"sqrt", "log2"} or int/float
        Features per split.
    feature_strategy : {"minimal", "core", "all"}
        Feature selection strategy.
    sample_for_fit : int or None
        Sample size (None = all).
    class_weight : str or None
        Class weighting strategy.
    random_state : int
        Random seed.
    top_n : int
        Top features to plot.
    save_plot : bool
        Whether to save plot.

    Returns
    -------
    dict
        Model, feature importance, OOB score, plot path.
    """
    data = clean_data().copy()
    y_full = data["voted"].astype(int).values

    # Select and prepare features
    feature_cols = select_features(data, feature_strategy)
    X_full, feature_names = prepare_features(data, feature_cols)

    # Subsample for fitting
    X_train,X_test,y_train,y_test = train_test_split(
                                                X_full,y_full,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=y_full)

    # Fit random forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=-1,
        oob_score=True,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_hat = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_logloss = log_loss(y_test, y_proba)
    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    train_logloss = log_loss(y_train, rf.predict_proba(X_train)[:,1])

    # Feature importance
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Plot importance
    plot_path = None
    if save_plot:
        paths = get_project_paths()
        top_imp = imp_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 0.4 * top_n + 2))
        ax.barh(top_imp.feature[::-1], top_imp.importance[::-1], color="steelblue")
        ax.set_xlabel("Importance (Gini decrease)")
        ax.set_title(
            f"Random Forest Feature Importance (top {top_n})\n"
            f"n_estimators={n_estimators}, max_depth={max_depth}, leaf≥{min_samples_leaf}"
        )
        
        plot_path = paths["plots"] + "random_forest_importance.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved importance plot to: {plot_path}")

    print(
        f"Random Forest fitted (strategy={feature_strategy}, "
        f"n_features={len(feature_names)}, n_samples={X_train.shape[0]}). "
        f"OOB score={rf.oob_score_:.4f}"
    )

    return {
        "model": rf,
        "feature_names": feature_names,
        "importance_df": imp_df,
        "oob_score": rf.oob_score_,
        "plot_path": plot_path,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_logloss": test_logloss,
    }
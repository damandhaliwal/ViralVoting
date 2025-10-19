from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score
from data_clean import clean_data
from utils import get_project_paths, select_features, prepare_features
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Literal, Optional
from sklearn.model_selection import train_test_split


def _build_intensity_vars(data):
    if {"cluster", "treatment_control"}.issubset(data.columns):
        cluster_stats = (
            data.groupby("cluster")["treatment_control"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "pct_control", "count": "cluster_size"})
            .reset_index()
        )
        cluster_stats["treatment_intensity"] = 1.0 - cluster_stats["pct_control"]
        median_intensity = cluster_stats["treatment_intensity"].median()
        data = data.merge(
            cluster_stats[["cluster", "treatment_intensity", "cluster_size"]],
            on="cluster",
            how="left",
        )
        if "treatment_intensity" in data.columns:
            data["high_cluster_intensity"] = (
                    data["treatment_intensity"] >= median_intensity
            ).astype(int)
    return data


def save_tree_plot(
        model,
        feature_names: list[str],
        title: str,
        filename: str,
        class_names: Optional[list[str]] = None,
        figsize: tuple[int, int] = (18, 10)
) -> str:
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
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def classification_tree(
        max_depth: int = 4,
        min_samples_leaf: int = 300,
        criterion: str = "gini",
        class_weight: str = "balanced",
        random_state: int = 42,
        feature_strategy: Literal["minimal", "core", "all"] = "core",
) -> dict:
    data = clean_data().copy()
    data = _build_intensity_vars(data)
    y_full = data["voted"].astype(int).values

    feature_cols = select_features(data, feature_strategy)
    for col in ["treatment_intensity", "high_cluster_intensity"]:
        if col in data.columns and col not in feature_cols:
            feature_cols.append(col)

    X_full, feature_names = prepare_features(data, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_full
    )

    clf = DTC(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_precision = precision_score(y_test, y_hat, zero_division=0)
    test_recall = recall_score(y_test, y_hat, zero_division=0)
    test_f1 = f1_score(y_test, y_hat, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_logloss = log_loss(y_test, y_proba)
    test_mse = np.mean((y_test - y_proba) ** 2)

    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    train_logloss = log_loss(y_train, clf.predict_proba(X_train)[:, 1])

    plot_path = save_tree_plot(
        clf,
        feature_names,
        "Classification Tree Predicting Voting (Yes/No)",
        "figure5.png",
        class_names=["No", "Yes"],
        figsize=(30, 16)
    )

    return {
        "model": clf,
        "feature_names": feature_names,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_logloss": test_logloss,
        "test_mse": test_mse,
        "plot_path": plot_path,
    }


def random_forest_classifier(
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 300,
        max_features: str = "sqrt",
        feature_strategy: Literal["minimal", "core", "all"] = "core",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        top_n: int = 20,
) -> dict:
    data = clean_data().copy()
    data = _build_intensity_vars(data)
    y_full = data["voted"].astype(int).values

    feature_cols = select_features(data, feature_strategy)
    for col in ["treatment_intensity", "high_cluster_intensity"]:
        if col in data.columns and col not in feature_cols:
            feature_cols.append(col)

    X_full, feature_names = prepare_features(data, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_full
    )

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

    y_hat = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_precision = precision_score(y_test, y_hat, zero_division=0)
    test_recall = recall_score(y_test, y_hat, zero_division=0)
    test_f1 = f1_score(y_test, y_hat, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_logloss = log_loss(y_test, y_proba)
    test_mse = np.mean((y_test - y_proba) ** 2)

    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    train_logloss = log_loss(y_train, rf.predict_proba(X_train)[:, 1])

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    paths = get_project_paths()
    top_imp = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 0.4 * top_n + 2))
    ax.barh(top_imp.feature[::-1], top_imp.importance[::-1], color="steelblue")
    ax.set_xlabel("Importance (Gini decrease)")
    ax.set_title(
        f"Random Forest Feature Importance (top {top_n})\n"
        f"n_estimators={n_estimators}, max_depth={max_depth}, leaf≥{min_samples_leaf}"
    )

    plot_path = paths["plots"] + "figure6.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": rf,
        "feature_names": feature_names,
        "importance_df": imp_df,
        "oob_score": rf.oob_score_,
        "plot_path": plot_path,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_logloss": test_logloss,
        "test_mse": test_mse,
    }


def bagging_classifier(
        n_estimators: int = 300,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 200,
        feature_strategy: Literal["minimal", "core", "all"] = "core",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
) -> dict:
    data = clean_data().copy()
    data = _build_intensity_vars(data)
    y_full = data["voted"].astype(int).values

    feature_cols = select_features(data, feature_strategy)
    for col in ["treatment_intensity", "high_cluster_intensity"]:
        if col in data.columns and col not in feature_cols:
            feature_cols.append(col)

    X_full, feature_names = prepare_features(data, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_full
    )

    base_estimator = DTC(
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

    y_hat = bagging.predict(X_test)
    y_proba = bagging.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_precision = precision_score(y_test, y_hat, zero_division=0)
    test_recall = recall_score(y_test, y_hat, zero_division=0)
    test_f1 = f1_score(y_test, y_hat, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_logloss = log_loss(y_test, y_proba)
    test_mse = np.mean((y_test - y_proba) ** 2)

    train_accuracy = accuracy_score(y_train, bagging.predict(X_train))
    train_logloss = log_loss(y_train, bagging.predict_proba(X_train)[:, 1])

    return {
        "model": bagging,
        "feature_names": feature_names,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_logloss": test_logloss,
        "test_mse": test_mse,
    }


def boosting_classifier(
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        feature_strategy: Literal["minimal", "core", "all"] = "core",
        random_state: int = 42,
) -> dict:
    data = clean_data().copy()
    data = _build_intensity_vars(data)
    y_full = data["voted"].astype(int).values

    feature_cols = select_features(data, feature_strategy)
    for col in ["treatment_intensity", "high_cluster_intensity"]:
        if col in data.columns and col not in feature_cols:
            feature_cols.append(col)

    X_full, feature_names = prepare_features(data, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_full
    )

    boosting = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    boosting.fit(X_train, y_train)

    y_hat = boosting.predict(X_test)
    y_proba = boosting.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_hat)
    test_precision = precision_score(y_test, y_hat, zero_division=0)
    test_recall = recall_score(y_test, y_hat, zero_division=0)
    test_f1 = f1_score(y_test, y_hat, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_logloss = log_loss(y_test, y_proba)
    test_mse = np.mean((y_test - y_proba) ** 2)

    train_accuracy = accuracy_score(y_train, boosting.predict(X_train))
    train_logloss = log_loss(y_train, boosting.predict_proba(X_train)[:, 1])

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": boosting.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": boosting,
        "feature_names": feature_names,
        "importance_df": imp_df,
        "train_accuracy": train_accuracy,
        "train_logloss": train_logloss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_logloss": test_logloss,
        "test_mse": test_mse,
    }


def run_model_comparison(
        feature_strategy: Literal["minimal", "core", "all"] = "core",
        random_state: int = 42
) -> dict:
    tree_results = classification_tree(
        feature_strategy=feature_strategy,
        random_state=random_state
    )

    forest_results = random_forest_classifier(
        feature_strategy=feature_strategy,
        random_state=random_state
    )

    bagging_results = bagging_classifier(
        feature_strategy=feature_strategy,
        random_state=random_state
    )

    boosting_results = boosting_classifier(
        feature_strategy=feature_strategy,
        random_state=random_state
    )

    comparison_data = [
        {
            "Model": "Tree",
            "Accuracy": tree_results["test_accuracy"],
            "Precision": tree_results["test_precision"],
            "Recall": tree_results["test_recall"],
            "F1": tree_results["test_f1"],
            "ROC AUC": tree_results["test_roc_auc"],
            "Log Loss": tree_results["test_logloss"],
            "MSE": tree_results["test_mse"],
        },
        {
            "Model": "Forest",
            "Accuracy": forest_results["test_accuracy"],
            "Precision": forest_results["test_precision"],
            "Recall": forest_results["test_recall"],
            "F1": forest_results["test_f1"],
            "ROC AUC": forest_results["test_roc_auc"],
            "Log Loss": forest_results["test_logloss"],
            "MSE": forest_results["test_mse"],
        },
        {
            "Model": "Bagging",
            "Accuracy": bagging_results["test_accuracy"],
            "Precision": bagging_results["test_precision"],
            "Recall": bagging_results["test_recall"],
            "F1": bagging_results["test_f1"],
            "ROC AUC": bagging_results["test_roc_auc"],
            "Log Loss": bagging_results["test_logloss"],
            "MSE": bagging_results["test_mse"],
        },
        {
            "Model": "Boosting",
            "Accuracy": boosting_results["test_accuracy"],
            "Precision": boosting_results["test_precision"],
            "Recall": boosting_results["test_recall"],
            "F1": boosting_results["test_f1"],
            "ROC AUC": boosting_results["test_roc_auc"],
            "Log Loss": boosting_results["test_logloss"],
            "MSE": boosting_results["test_mse"],
        },
    ]

    comparison_df = pd.DataFrame(comparison_data)

    paths = get_project_paths()

    latex_table = comparison_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Model Comparison: Tree, Forest, Bagging, Boosting",
        label="tab:model_comparison",
    )

    output_path = paths["tables"] + "table5.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    return {
        "comparison_df": comparison_df,
        "tree_results": tree_results,
        "forest_results": forest_results,
        "bagging_results": bagging_results,
        "boosting_results": boosting_results,
        "latex_path": output_path
    }
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_clean import clean_data
from utils import get_project_paths, prepare_features, select_features


def save_tree_plot(
    model,
    feature_names: list[str],
    title: str,
    filename: str,
    class_names: Optional[list[str]] = None,
    figsize: tuple[int, int] = (18, 10),
    max_depth: Optional[int] = None,
) -> str:
    """Replicate tree_forest plotting utility with 600 dpi output."""
    paths = get_project_paths()
    fig, ax = plt.subplots(figsize=figsize)

    from sklearn.tree import plot_tree

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        impurity=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        ax=ax,
    )
    ax.set_title(title)

    plot_path = paths["plots"] + filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {plot_path}")
    return plot_path


def _evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_proba = 1.0 / (1.0 + np.exp(-decision))
    else:
        y_proba = y_pred.astype(float)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float("nan")

    try:
        test_logloss = log_loss(y_test, y_proba)
    except ValueError:
        test_logloss = float("nan")
    
    # MSE (Brier Score) for classification - no square root
    mse = float(np.mean((y_test - y_proba) ** 2))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "log_loss": test_logloss,
        "mse": mse,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def run_diffusion_ml(test_size: float = 0.3, random_state: int = 42):
    data = clean_data()
    data = data.dropna(subset=["voted"]).copy()

    # Build cluster intensity measures mirroring the diffusion study.
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
        data["high_cluster_intensity"] = (
            data["treatment_intensity"] >= median_intensity
        ).astype(int)
    else:
        data["treatment_intensity"] = np.nan
        data["high_cluster_intensity"] = np.nan

    feature_cols = select_features(data, strategy="core")
    if not feature_cols:
        feature_cols = select_features(data, strategy="minimal")
    if not feature_cols:
        feature_cols = select_features(data, strategy="all")
    if not feature_cols:
        raise ValueError("No usable features found for modelling.")

    for col in ["treatment_intensity", "high_cluster_intensity"]:
        if col in data.columns and col not in feature_cols:
            feature_cols.append(col)

    X, feature_names = prepare_features(data, feature_cols)
    y = data["voted"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    records = []
    feature_importances = {}

    # Classification tree mirroring tree_forest configuration.
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight=None,
        random_state=random_state,
    )
    tree_metrics = _evaluate_model(tree_model, X_train, X_test, y_train, y_test)

    tree_train_pred = tree_model.predict(X_train)
    tree_train_proba = tree_model.predict_proba(X_train)[:, 1]
    tree_train_accuracy = accuracy_score(y_train, tree_train_pred)
    try:
        tree_train_logloss = log_loss(y_train, tree_train_proba)
    except ValueError:
        tree_train_logloss = float("nan")

    records.append(
        {
            "model": "classification_tree",
            "accuracy": tree_metrics["accuracy"],
            "precision": tree_metrics["precision"],
            "recall": tree_metrics["recall"],
            "f1": tree_metrics["f1"],
            "roc_auc": tree_metrics["roc_auc"],
            "log_loss": tree_metrics["log_loss"],
            "mse": tree_metrics["mse"],
        }
    )

    tree_plot_path = save_tree_plot(
        tree_model,
        feature_names,
        "Classification Tree Predicting Voting (Yes/No)",
        "diffusion_classification_tree.png",
        class_names=["No", "Yes"],
        figsize=(30, 16),
        max_depth=4,
    )

    feature_importances["classification_tree"] = (
        pd.Series(tree_model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )

    # Bagging ensemble mirroring bagging.py defaults.
    bagging_base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=random_state,
    )
    bagging_model = BaggingClassifier(
        estimator=bagging_base,
        n_estimators=300,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )
    bagging_metrics = _evaluate_model(bagging_model, X_train, X_test, y_train, y_test)

    records.append(
        {
            "model": "bagging",
            "accuracy": bagging_metrics["accuracy"],
            "precision": bagging_metrics["precision"],
            "recall": bagging_metrics["recall"],
            "f1": bagging_metrics["f1"],
            "roc_auc": bagging_metrics["roc_auc"],
            "log_loss": bagging_metrics["log_loss"],
            "mse": bagging_metrics["mse"],
        }
    )

    # Random forest with tree_forest hyper-parameters.
    rf_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        n_jobs=-1,
        oob_score=True,
        random_state=random_state,
    )
    rf_metrics = _evaluate_model(rf_model, X_train, X_test, y_train, y_test)

    rf_train_pred = rf_model.predict(X_train)
    rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
    try:
        rf_train_logloss = log_loss(y_train, rf_train_proba)
    except ValueError:
        rf_train_logloss = float("nan")

    records.append(
        {
            "model": "random_forest",
            "accuracy": rf_metrics["accuracy"],
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f1": rf_metrics["f1"],
            "roc_auc": rf_metrics["roc_auc"],
            "log_loss": rf_metrics["log_loss"],
            "mse": rf_metrics["mse"],
        }
    )

    rf_importance = (
        pd.Series(rf_model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
    feature_importances["random_forest"] = rf_importance

    # Gradient boosting on the full sample.
    boosting_model_full = GradientBoostingClassifier(random_state=random_state)
    boosting_metrics_full = _evaluate_model(
        boosting_model_full, X_train, X_test, y_train, y_test
    )

    records.append(
        {
            "model": "boosting",
            "accuracy": boosting_metrics_full["accuracy"],
            "precision": boosting_metrics_full["precision"],
            "recall": boosting_metrics_full["recall"],
            "f1": boosting_metrics_full["f1"],
            "roc_auc": boosting_metrics_full["roc_auc"],
            "log_loss": boosting_metrics_full["log_loss"],
            "mse": boosting_metrics_full["mse"],
        }
    )

    if hasattr(boosting_model_full, "feature_importances_"):
        feature_importances["boosting"] = (
            pd.Series(boosting_model_full.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
        )

    # Focused boosting on control group members in high intensity clusters.
    boosting_record = None
    if {"treatment_control", "high_cluster_intensity"}.issubset(data.columns):
        mask_control_high = (
            (data["treatment_control"] == 1)
            & (data["high_cluster_intensity"] == 1)
        )
        if mask_control_high.sum() > 10:
            X_sub = X[mask_control_high.values]
            y_sub = y[mask_control_high.values]

            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_sub,
                y_sub,
                test_size=test_size,
                stratify=y_sub if y_sub.sum() not in {0, len(y_sub)} else None,
                random_state=random_state,
            )

            boosting_model = GradientBoostingClassifier(random_state=random_state)
            metrics = _evaluate_model(
                boosting_model,
                X_train_sub,
                X_test_sub,
                y_train_sub,
                y_test_sub,
            )
            boosting_record = {
                "model": "boosting_control_high",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "mse": metrics["mse"],
            }
            records.append(boosting_record)

            if hasattr(boosting_model, "feature_importances_"):
                ranking = (
                    pd.Series(boosting_model.feature_importances_, index=feature_names)
                    .sort_values(ascending=False)
                )
                feature_importances["boosting_control_high"] = ranking

    results_df = pd.DataFrame(records)
    metric_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "mse",
    ]
    results_df = results_df[metric_columns].sort_values(
        by="roc_auc", ascending=False
    )

    paths = get_project_paths()
    latex_table = results_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Machine Learning Models for Information Diffusion",
        label="tab:diffusion_ml",
    )

    output_path = paths["tables"] + "diffusion_ml_results.tex"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(latex_table)

    name_map = {
        "classification_tree": "Tree",
        "random_forest": "Forest",
        "bagging": "Bagging",
        "boosting": "Boosting",
    }
    comparison_order = list(name_map.keys())
    comparison_df = (
        results_df[results_df["model"].isin(comparison_order)]
        .set_index("model")
        .reindex(comparison_order)
        .dropna(how="all")
        .reset_index()
    )
    comparison_df["Model"] = comparison_df["model"].map(name_map)
    comparison_display = comparison_df[
        [
            "Model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "mse",
        ]
    ].rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC AUC",
            "log_loss": "Log Loss",
            "mse": "MSE",
        }
    )

    comparison_output_path = paths["tables"] + "diffusion_ml_model_comparison.tex"
    comparison_display.to_latex(
        comparison_output_path,
        index=False,
        float_format="%.3f",
        caption="Model Comparison: Tree, Forest, Bagging, Boosting",
        label="tab:diffusion_ml_comparison",
    )

    print("Model performance summary:")
    print(results_df)
    print(f"\nSaved LaTeX table to: {output_path}")
    print("\nComparison table (four core methods):")
    print(comparison_display)
    print(f"\nSaved comparison table to: {comparison_output_path}")

    if feature_importances:
        top_contributors = pd.concat(
            {
                name: series.head(10)
                for name, series in feature_importances.items()
            },
            axis=1,
        )
        csv_path = paths["tables"] + "diffusion_ml_feature_importance.csv"
        top_contributors.to_csv(csv_path)
        print(f"Saved feature importance snapshots to: {csv_path}")

    # Visualize the decision tree structure.
    # Tree plot already saved via save_tree_plot.

    # Heatmap for top random forest predictors.
    if "random_forest" in feature_importances:
        top_rf = feature_importances["random_forest"].head(10).to_frame("importance")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(top_rf))))
        sns.heatmap(
            top_rf.sort_values("importance", ascending=True),
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_title("Top Random Forest Predictors")
        plt.tight_layout()
        rf_path = paths["plots"] + "random_forest_feature_importance.png"
        plt.savefig(rf_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved random forest importance heatmap to: {rf_path}")

    return {
        "results": results_df,
        "feature_importances": feature_importances,
        "boosting_control_high": boosting_record,
        "comparison_table": comparison_output_path,
    }


if __name__ == "__main__":
    run_diffusion_ml()
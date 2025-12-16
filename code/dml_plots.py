# Cumulative Gain Plots for Double ML
# Daman Dhaliwal
import os
# import libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.linear_model import LogisticRegression, LassoCV
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier

from data_clean import clean_data
from utils import get_project_paths


def estimate_linear_dml(Y, T, X):
    dml = LinearDML(
        model_y=LassoCV(cv=5, max_iter=2000),
        model_t=LogisticRegression(C=1e6, max_iter=1000),
        discrete_treatment=True,
        cv=5,
        random_state=42
    )
    dml.fit(Y, T, X=X)
    return dml


def estimate_causal_forest_dml(Y, T, X):
    dml = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200, max_depth=5, random_state=42, verbose=-1),
        model_t=LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1),
        discrete_treatment=True,
        cv=5,
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    dml.fit(Y, T, X=X)
    return dml


def compute_cumulative_gain(cate, Y, T):
    # Sort by predicted CATE (descending - highest predicted effect first)
    sorted_idx = np.argsort(-cate)
    Y_sorted = Y[sorted_idx]
    T_sorted = T[sorted_idx]

    n = len(Y)
    percentiles = np.linspace(0.01, 1.0, 100)

    cumulative_gain = []
    random_gain = []

    # Overall ATE (simple difference in means)
    ate_overall = Y[T == 1].mean() - Y[T == 0].mean()

    for p in percentiles:
        i = int(p * n)
        if i == 0:
            i = 1

        top_Y = Y_sorted[:i]
        top_T = T_sorted[:i]

        # ATE in top fraction
        if top_T.sum() > 0 and (top_T == 0).sum() > 0:
            top_ate = top_Y[top_T == 1].mean() - top_Y[top_T == 0].mean()
        else:
            top_ate = ate_overall

        cumulative_gain.append(p * top_ate)
        random_gain.append(p * ate_overall)

    return np.array(cumulative_gain), np.array(random_gain), percentiles


def run_dml_cumulative_gain_plots():
    data = clean_data()
    paths = get_project_paths()

    FEATURES = ["sex", "yob", "g2000", "g2002", "g2004", "p2000", "p2002",
        "treatment_civic duty", "treatment_hawthorne", "treatment_neighbors",
        "treatment_self", "voted", "treatment_intensity"]
    data = data[FEATURES]

    covariate_cols = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002', 'treatment_intensity']
    Y_col = 'voted'

    treatments = {
        'Civic Duty': 'treatment_civic duty',
        'Hawthorne': 'treatment_hawthorne',
        'Self': 'treatment_self',
        'Neighbors': 'treatment_neighbors'
    }

    linear_cates = {}
    cf_cates = {}
    Y_data = {}
    T_data = {}

    for name, treat_col in treatments.items():
        other_treatments = [t for t in treatments.values() if t != treat_col]
        data_temp = data[~data[other_treatments].any(axis=1)].copy()

        X = data_temp[covariate_cols].values
        Y = data_temp[Y_col].values
        T = data_temp[treat_col].values

        # Linear DML
        linear_dml = estimate_linear_dml(Y, T, X)
        linear_cates[name] = linear_dml.effect(X)

        # Causal Forest DML
        cf_dml = estimate_causal_forest_dml(Y, T, X)
        cf_cates[name] = cf_dml.effect(X)

        Y_data[name] = Y
        T_data[name] = T


    # Plot: 4 subplots - one per treatment, linear vs CF on same subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    treatment_names = list(treatments.keys())

    for idx, name in enumerate(treatment_names):
        ax = axes[idx]

        # Linear DML
        linear_gain, random_gain, percentiles = compute_cumulative_gain(
            linear_cates[name], Y_data[name], T_data[name]
        )

        # Causal Forest DML
        cf_gain, _, _ = compute_cumulative_gain(
            cf_cates[name], Y_data[name], T_data[name]
        )

        ax.plot(percentiles, linear_gain, 'b-', linewidth=2, label='Linear DML')
        ax.plot(percentiles, cf_gain, 'r-', linewidth=2, label='Causal Forest DML')
        ax.plot(percentiles, random_gain, 'k--', linewidth=1.5, label='Random', alpha=0.7)

        ax.set_xlabel('Proportion Targeted', fontsize=10)
        ax.set_ylabel('Cumulative Gain', fontsize=10)
        ax.set_title(f'{name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cumulative Gain Plots: Linear vs Causal Forest DML', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = paths['plots']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'figure8.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    return


if __name__ == "__main__":
    run_dml_cumulative_gain_plots()
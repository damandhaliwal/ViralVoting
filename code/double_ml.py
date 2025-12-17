# Double Machine Learning (DML) Analysis
# Daman Dhaliwal

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier

from data_clean import clean_data
from utils import get_project_paths


def estimate_linear_dml(Y, T, X, model_y=None, model_t=None):
    if model_y is None:
        model_y = LassoCV(cv=5, max_iter=2000)
    if model_t is None:
        model_t = LogisticRegression(C=1e6, max_iter=1000)

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=5,
        random_state=42
    )
    dml.fit(Y, T, X=X)

    ate = dml.ate(X)

    return dml, ate


def estimate_causal_forest_dml(Y, T, X, model_y=None, model_t=None):
    if model_y is None:
        model_y = LGBMRegressor(n_estimators=200, max_depth=5, random_state=42, verbose=-1)
    if model_t is None:
        model_t = LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1)

    dml = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=5,
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    dml.fit(Y, T, X=X)

    ate = dml.ate(X)

    return dml, ate


def bootstrap_ate(dml_model, X, n_bootstrap=500):
    np.random.seed(42)
    cate = dml_model.effect(X)

    ate_bootstrap = []
    n = len(X)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        ate_boot = np.mean(cate[idx])
        ate_bootstrap.append(ate_boot)

    se = np.std(ate_bootstrap)
    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)

    return se, ci_lower, ci_upper


def compute_cumulative_gain_intensity(cate, Y, T, intensity):
    sorted_idx = np.argsort(-intensity)
    Y_sorted = Y[sorted_idx]
    T_sorted = T[sorted_idx]
    cate_sorted = cate[sorted_idx]

    n = len(Y)
    cumulative_gain = []
    random_gain = []

    ate_overall = np.mean(cate)

    percentiles = np.linspace(0.01, 1.0, 100)

    for p in percentiles:
        i = int(p * n)
        if i == 0:
            i = 1

        top_cate = cate_sorted[:i]
        cumulative_gain.append(p * np.mean(top_cate))
        random_gain.append(p * ate_overall)

    return np.array(cumulative_gain), np.array(random_gain), percentiles


def plot_cumulative_gain_intensity(linear_cate, cf_cate, Y, T, intensity, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    linear_gain, random_gain, percentiles = compute_cumulative_gain_intensity(
        linear_cate, Y, T, intensity
    )
    cf_gain, _, _ = compute_cumulative_gain_intensity(
        cf_cate, Y, T, intensity
    )

    ax.plot(percentiles, linear_gain, 'b-', linewidth=2, label='Linear DML')
    ax.plot(percentiles, cf_gain, 'r-', linewidth=2, label='Causal Forest DML')
    ax.plot(percentiles, random_gain, 'k--', linewidth=1.5, label='Random Targeting')

    ax.set_xlabel('Proportion of Population (by Treatment Intensity)', fontsize=12)
    ax.set_ylabel('Cumulative Gain', fontsize=12)
    ax.set_title('Cumulative Gain: Treatment Intensity Effect', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    return fig


def run_double_ml_analysis():
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

    paper_results = {
        'Civic Duty': 0.0180,
        'Hawthorne': 0.0260,
        'Self': 0.0490,
        'Neighbors': 0.0810
    }

    all_results = []

    for name, treat_col in treatments.items():
        other_treatments = [t for t in treatments.values() if t != treat_col]
        data_temp = data[~data[other_treatments].any(axis=1)].copy()

        X = data_temp[covariate_cols].values
        Y = data_temp[Y_col].values
        T = data_temp[treat_col].values

        n_treated = T.sum()
        n_control = (T == 0).sum()

        linear_dml, linear_ate = estimate_linear_dml(Y, T, X)
        linear_cate = linear_dml.effect(X)
        linear_se, linear_ci_low, linear_ci_high = bootstrap_ate(linear_dml, X, n_bootstrap=500)

        cf_dml, cf_ate = estimate_causal_forest_dml(Y, T, X)
        cf_cate = cf_dml.effect(X)
        cf_se, cf_ci_low, cf_ci_high = bootstrap_ate(cf_dml, X, n_bootstrap=500)

        cate_correlation = np.corrcoef(linear_cate, cf_cate)[0, 1]
        cate_mae = np.mean(np.abs(linear_cate - cf_cate))

        all_results.append({
            'Treatment': name,
            'Linear ATE': linear_ate,
            'Linear SE': linear_se,
            'Linear CI Lower': linear_ci_low,
            'Linear CI Upper': linear_ci_high,
            'Linear CATE Std': np.std(linear_cate),
            'CF ATE': cf_ate,
            'CF SE': cf_se,
            'CF CI Lower': cf_ci_low,
            'CF CI Upper': cf_ci_high,
            'CF CATE Std': np.std(cf_cate),
            'CATE Correlation': cate_correlation,
            'CATE MAE': cate_mae,
            'Paper ATE': paper_results[name]
        })

    results_df = pd.DataFrame(all_results)

    output_path = paths['tables'] + 'double_ml_comparison.csv'
    results_df.to_csv(output_path, index=False)

    run_intensity_analysis(data, treatments, covariate_cols, Y_col, paths)

    return results_df


def run_intensity_analysis(data, treatments, covariate_cols, Y_col, paths):
    intensity_tertiles = data['treatment_intensity'].quantile([0, 1/3, 2/3, 1])
    data['intensity_level'] = pd.cut(
        data['treatment_intensity'],
        bins=intensity_tertiles,
        labels=['Low', 'Mid', 'High'],
        include_lowest=True
    )

    X_covariates = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']
    intensity_levels = ['Low', 'Mid', 'High']

    intensity_results = []

    for intensity in intensity_levels:
        intensity_data = data[data['intensity_level'] == intensity].copy()

        for treatment_name, treatment_col in treatments.items():
            other_treatments = [t for t in treatments.values() if t != treatment_col]
            subset_data = intensity_data[~intensity_data[other_treatments].any(axis=1)].copy()

            n_treated = subset_data[treatment_col].sum()
            n_control = (subset_data[treatment_col] == 0).sum()

            if n_treated < 50 or n_control < 50:
                continue

            X = subset_data[X_covariates].values
            Y = subset_data[Y_col].values
            T = subset_data[treatment_col].values

            try:
                linear_dml, linear_ate = estimate_linear_dml(Y, T, X)
                linear_se, linear_ci_low, linear_ci_high = bootstrap_ate(linear_dml, X, n_bootstrap=200)
            except:
                linear_ate = np.nan
                linear_se, linear_ci_low, linear_ci_high = np.nan, np.nan, np.nan

            try:
                cf_dml, cf_ate = estimate_causal_forest_dml(Y, T, X)
                cf_se, cf_ci_low, cf_ci_high = bootstrap_ate(cf_dml, X, n_bootstrap=200)
            except:
                cf_ate = np.nan
                cf_se, cf_ci_low, cf_ci_high = np.nan, np.nan, np.nan

            intensity_results.append({
                'Intensity Level': intensity,
                'Treatment': treatment_name,
                'Linear ATE': linear_ate,
                'Linear SE': linear_se,
                'Linear CI Lower': linear_ci_low,
                'Linear CI Upper': linear_ci_high,
                'CF ATE': cf_ate,
                'CF SE': cf_se,
                'CF CI Lower': cf_ci_low,
                'CF CI Upper': cf_ci_high,
                'N Treated': int(n_treated),
                'N Control': int(n_control)
            })

    intensity_df = pd.DataFrame(intensity_results)

    output_path = paths['tables'] + 'table12.tex'
    intensity_df.to_latex(output_path, index=False, float_format='%.4f')

    run_cumulative_gain_analysis(data, Y_col, paths)

    return intensity_df


def run_cumulative_gain_analysis(data, Y_col, paths):
    data['any_treatment'] = (
        data['treatment_civic duty'] |
        data['treatment_hawthorne'] |
        data['treatment_self'] |
        data['treatment_neighbors']
    ).astype(int)

    X_intensity = data[['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']].values
    Y_intensity = data[Y_col].values
    T_intensity = data['any_treatment'].values
    intensity_values = data['treatment_intensity'].values

    linear_dml_intensity, linear_ate_intensity = estimate_linear_dml(Y_intensity, T_intensity, X_intensity)
    linear_cate_intensity = linear_dml_intensity.effect(X_intensity)

    cf_dml_intensity, cf_ate_intensity = estimate_causal_forest_dml(Y_intensity, T_intensity, X_intensity)
    cf_cate_intensity = cf_dml_intensity.effect(X_intensity)

    output_dir = paths['plots']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'cumulative_gain_treatment_intensity.png'
    plot_cumulative_gain_intensity(
        linear_cate_intensity,
        cf_cate_intensity,
        Y_intensity,
        T_intensity,
        intensity_values,
        save_path=output_path
    )

    intensity_quantiles = pd.qcut(intensity_values, q=5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])

    cate_by_quantile = []
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
        mask = intensity_quantiles == q
        cate_by_quantile.append({
            'Quantile': q,
            'Linear CATE': np.mean(linear_cate_intensity[mask]),
            'CF CATE': np.mean(cf_cate_intensity[mask]),
            'N': mask.sum()
        })

    cate_quantile_df = pd.DataFrame(cate_by_quantile)

    output_path = paths['tables'] + 'cate_by_intensity_quantile.csv'
    cate_quantile_df.to_csv(output_path, index=False)

    return cate_quantile_df


if __name__ == "__main__":
    run_double_ml_analysis()
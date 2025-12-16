# Doubly Robust Estimation for Treatment Effect Intensity
# Daman Dhaliwal

# import libraries
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from data_clean import clean_data
from utils import get_project_paths


data = clean_data()
FEATURES = ["sex", "yob", "g2000", "g2002", "g2004", "p2000", "p2002",
    "treatment_civic duty", "treatment_hawthorne", "treatment_neighbors",
    "treatment_self", "voted", "treatment_intensity"]
data = data[FEATURES]


def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]

    treated = df[T] == 1
    control = df[T] == 0

    mu0 = LinearRegression().fit(df.loc[control, X], df.loc[control, Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.loc[treated, X], df.loc[treated, Y]).predict(df[X])

    return (
            np.mean(df[T] * (df[Y] - mu1) / ps + mu1) -
            np.mean((1 - df[T]) * (df[Y] - mu0) / (1 - ps) + mu0)
    )

def bootstrap_ci(df, X, T, Y, n_bootstrap=1000):
    ate_bootstrap = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        boot_sample = df.sample(n=len(df), replace=True)
        ate_boot = doubly_robust(boot_sample, X, T, Y)
        ate_bootstrap.append(ate_boot)

    se = np.std(ate_bootstrap)
    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)

    return ate_bootstrap, se, ci_lower, ci_upper

def run_dr_intensity_analysis():
    # Create three-level intensity variable using tertiles
    intensity_tertiles = data['treatment_intensity'].quantile([0, 1/3, 2/3, 1])
    data['intensity_level'] = pd.cut(
        data['treatment_intensity'],
        bins=intensity_tertiles,
        labels=['Low', 'Mid', 'High'],
        include_lowest=True
    )

    # Define covariates
    X_covariates = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']
    Y_outcome = 'voted'

    # Define treatments
    treatments = {
        'Civic Duty': 'treatment_civic duty',
        'Hawthorne': 'treatment_hawthorne',
        'Self': 'treatment_self',
        'Neighbors': 'treatment_neighbors'
    }

    # Intensity levels
    intensity_levels = ['Low', 'Mid', 'High']

    # Store all results
    all_results = []


    for intensity in intensity_levels:
        intensity_data = data[data['intensity_level'] == intensity].copy()

        for treatment_name, treatment_col in treatments.items():
            other_treatments = [t for t in treatments.values() if t != treatment_col]
            subset_data = intensity_data[~intensity_data[other_treatments].any(axis=1)].copy()

            n_treated = subset_data[treatment_col].sum()
            n_control = (subset_data[treatment_col] == 0).sum()
            n_total = len(subset_data)

            if n_treated < 50 or n_control < 50:
                print("Skipping: insufficient sample size")
                continue

            # Calculate turnout rates
            control_turnout = subset_data[subset_data[treatment_col] == 0]['voted'].mean()
            treated_turnout = subset_data[subset_data[treatment_col] == 1]['voted'].mean()

            # Calculate doubly robust ATE
            ate_dr = doubly_robust(subset_data, X_covariates, treatment_col, Y_outcome)
            _, se_dr, ci_lower, ci_upper = bootstrap_ci(
                subset_data, X_covariates, treatment_col, Y_outcome, n_bootstrap=500
            )

            # Store results
            all_results.append({
                'Intensity Level': intensity,
                'Treatment': treatment_name,
                'Control Turnout': control_turnout,
                'Treated Turnout': treated_turnout,
                'DR ATE': ate_dr,
                'SE': se_dr,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'N Treated': int(n_treated),
                'N Control': int(n_control),
                'N Total': n_total
            })

    results_df = pd.DataFrame(all_results)

    paths = get_project_paths()

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'table11.tex'

    results_df.to_latex(output_path, index=False, float_format="%.4f")
    return

if __name__ == "__main__":
    run_dr_intensity_analysis()
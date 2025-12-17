# Doubly Robust Estimation with Interaction Terms
# Daman Dhaliwal

# import libraries
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from data_clean import clean_data
from utils import get_project_paths


# define the doubly_robust function
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

# calculate bootstap CIs
def bootstrap_ci(df, X, T, Y, n_bootstrap = 100):
    ate_bootstrap = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        boot_sample = df.sample(n=len(df), replace=True)
        ate_boot = doubly_robust(boot_sample, X, T, Y)
        ate_bootstrap.append(ate_boot)

    # Calculate standard error and confidence interval
    se = np.std(ate_bootstrap)
    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)

    return ate_bootstrap, se, ci_lower, ci_upper


def generate_latex_table(results_df, paths, filename='table10.tex'):
    # Format columns for display
    table_df = results_df[['Treatment']].copy()

    # DR ATE with CI
    table_df['DR ATE'] = results_df.apply(
        lambda r: f"{r['DR ATE']:.4f}\n({r['CI Lower']:.4f}, {r['CI Upper']:.4f})", axis=1
    )

    # Interaction Effect with CI
    table_df['Interaction Effect'] = results_df.apply(
        lambda r: f"{r['Interaction Effect']:.4f}\n({r['Inter CI Lower']:.4f}, {r['Inter CI Upper']:.4f})", axis=1
    )

    latex_output = table_df.to_latex(
        index=False,
        escape=False,
        column_format='lcc',
        float_format='%.4f'
    )

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + filename
    with open(output_path, 'w') as f:
        f.write(latex_output)

    return latex_output


def run_dr_with_interactions():
    data = clean_data()
    paths = get_project_paths()

    treatments = {
        'Civic Duty': 'treatment_civic duty',
        'Hawthorne': 'treatment_hawthorne',
        'Self': 'treatment_self',
        'Neighbors': 'treatment_neighbors'
    }

    # Base covariates
    base_covariates = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002', 'treatment_intensity']
    Y = 'voted'

    # Create interaction terms
    interaction_vars = []
    for name, treat_col in treatments.items():
        inter_name = f'{treat_col}_x_intensity'
        data[inter_name] = data[treat_col] * data['treatment_intensity']
        interaction_vars.append(inter_name)

    # Store results
    all_results = []

    for name, treat_col in treatments.items():
        other_treatments = [t for t in treatments.values() if t != treat_col]
        data_temp = data[~data[other_treatments].any(axis=1)].copy()

        # Get the interaction term for this treatment
        inter_col = f'{treat_col}_x_intensity'

        # Covariates include base + this treatment's interaction
        X_cols = base_covariates + [inter_col]

        # Calculate sample sizes
        n_treated = data_temp[treat_col].sum()
        n_control = (data_temp[treat_col] == 0).sum()

        # Main Treatment Effect (DR) ---
        ate_dr = doubly_robust(data_temp, X_cols, treat_col, Y)
        _, se_dr, ci_lower, ci_upper = bootstrap_ci(
            data_temp, X_cols, treat_col, Y, n_bootstrap=100
        )

        # Estimate Interaction Effect
        # Split by median intensity to estimate differential effect
        median_intensity = data_temp['treatment_intensity'].median()

        # Low intensity subsample
        low_intensity = data_temp[data_temp['treatment_intensity'] <= median_intensity].copy()
        # High intensity subsample
        high_intensity = data_temp[data_temp['treatment_intensity'] > median_intensity].copy()

        # Estimate ATE in each subsample
        if len(low_intensity) > 100 and len(high_intensity) > 100:
            ate_low = doubly_robust(low_intensity, base_covariates, treat_col, Y)
            ate_high = doubly_robust(high_intensity, base_covariates, treat_col, Y)

            # Interaction effect = difference in ATEs
            interaction_effect = ate_high - ate_low

            # Bootstrap the interaction effect
            np.random.seed(42)
            inter_bootstrap = []
            for _ in range(100):
                boot_low = low_intensity.sample(n=len(low_intensity), replace=True)
                boot_high = high_intensity.sample(n=len(high_intensity), replace=True)

                ate_low_boot = doubly_robust(boot_low, base_covariates, treat_col, Y)
                ate_high_boot = doubly_robust(boot_high, base_covariates, treat_col, Y)

                inter_bootstrap.append(ate_high_boot - ate_low_boot)

            inter_se = np.std(inter_bootstrap)
            inter_ci_lower = np.percentile(inter_bootstrap, 2.5)
            inter_ci_upper = np.percentile(inter_bootstrap, 97.5)
        else:
            interaction_effect = np.nan
            inter_se = np.nan
            inter_ci_lower = np.nan
            inter_ci_upper = np.nan

        # Store results
        all_results.append({
            'Treatment': name,
            'DR ATE': ate_dr,
            'SE': se_dr,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'Interaction Effect': interaction_effect,
            'Inter SE': inter_se,
            'Inter CI Lower': inter_ci_lower,
            'Inter CI Upper': inter_ci_upper,
            'N Treated': int(n_treated),
            'N Control': int(n_control)
        })

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Generate LaTeX table
    generate_latex_table(results_df, paths)

    return results_df


if __name__ == "__main__":
    results = run_dr_with_interactions()

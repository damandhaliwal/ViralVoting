# Combined IPW and PSM Analysis using Multinomial Propensity Scores
# Daman Dhaliwal

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from data_clean import clean_data
from utils import get_project_paths

plt.style.use('fivethirtyeight')


class MultinomialPropensityScore:
    def __init__(self, data, outcome, treatment, confounders):
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.confounders = confounders

        self.X = data[confounders].values
        self.y = data[outcome].values
        self.t = data[treatment].values
        self.treatment_values = np.unique(self.t)

    def estimate_propensity_scores(self):
        self.ps_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.ps_model.fit(self.X, self.t)
        self.ps_scores = self.ps_model.predict_proba(self.X)
        return self.ps_scores

    def ipw_ate(self, trim_percentile=99):
        if not hasattr(self, 'ps_scores'):
            self.estimate_propensity_scores()

        # Calculate stabilized weights
        weights = np.zeros(len(self.t))
        for i in range(len(self.t)):
            treat_idx = np.where(self.treatment_values == self.t[i])[0][0]
            weights[i] = 1 / self.ps_scores[i, treat_idx]

        treat_probs = np.array([np.mean(self.t == val) for val in self.treatment_values])
        for i in range(len(self.t)):
            treat_idx = np.where(self.treatment_values == self.t[i])[0][0]
            weights[i] *= treat_probs[treat_idx]

        weight_max = np.percentile(weights, trim_percentile)
        weights = np.clip(weights, 0, weight_max)
        self.ipw_weights = weights

        control_value = 0
        results = {}

        for treat_val in [1, 2, 3, 4]:
            mask_treatment = (self.t == treat_val)
            mask_control = (self.t == control_value)

            y_treat_weighted = np.average(self.y[mask_treatment], weights=weights[mask_treatment])
            y_control_weighted = np.average(self.y[mask_control], weights=weights[mask_control])

            ate = y_treat_weighted - y_control_weighted
            results[treat_val] = ate

        return results

    def psm_ate(self, n_neighbors=1, caliper=0.005):
        if not hasattr(self, 'ps_scores'):
            self.estimate_propensity_scores()

        control_value = 0
        results = {}

        mask_control = (self.t == control_value)

        for treat_val in [1, 2, 3, 4]:
            # Extract propensity score for THIS specific treatment
            # Column index for this treatment
            treat_idx = np.where(self.treatment_values == treat_val)[0][0]

            ps_current = self.ps_scores[:, treat_idx]

            ps_treated = ps_current[self.t == treat_val].reshape(-1, 1)
            ps_control = ps_current[self.t == control_value].reshape(-1, 1)

            y_treated = self.y[self.t == treat_val]
            y_control = self.y[self.t == control_value]

            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
            nn.fit(ps_control)

            distances, indices = nn.kneighbors(ps_treated)

            valid_matches = distances[:, 0] <= caliper

            if valid_matches.sum() > 0:
                matched_treated = y_treated[valid_matches]
                matched_control = y_control[indices[valid_matches, 0]]
                ate = np.mean(matched_treated - matched_control)
            else:
                ate = np.nan

            results[treat_val] = ate

        return results


class TreatmentIntensityAnalysis:
    def __init__(self, data, outcome, intensity_var, confounders):
        self.data = data.copy()
        self.outcome = outcome
        self.intensity_var = intensity_var
        self.confounders = confounders
        self.X = data[confounders].values
        self.y = data[outcome].values
        self.intensity = data[intensity_var].values

    def analyze(self):
        # Binary Propensity Score for Intensity (High vs Low)
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(self.X, self.intensity)
        ps_scores = ps_model.predict_proba(self.X)[:, 1]

        # IPW
        weights = np.where(self.intensity == 1, 1 / ps_scores, 1 / (1 - ps_scores))
        weights = np.clip(weights, 0, np.percentile(weights, 99))

        mask_high = (self.intensity == 1)
        mask_low = (self.intensity == 0)

        ate_ipw = (np.average(self.y[mask_high], weights=weights[mask_high]) -
                   np.average(self.y[mask_low], weights=weights[mask_low]))

        # Simple analytic SE for IPW
        n1 = mask_high.sum()
        n0 = mask_low.sum()
        var1 = np.var(self.y[mask_high]) / n1
        var0 = np.var(self.y[mask_low]) / n0
        se_ipw = np.sqrt(var1 + var0)

        # PSM
        ps_scores_reshaped = ps_scores.reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(ps_scores_reshaped[mask_low])
        distances, indices = nn.kneighbors(ps_scores_reshaped[mask_high])

        valid_matches = distances[:, 0] <= 0.005
        matched_treated = self.y[mask_high][valid_matches]
        matched_control = self.y[mask_low][indices[valid_matches, 0]]

        if len(matched_treated) > 0:
            ate_psm = np.mean(matched_treated - matched_control)
            se_psm = np.std(matched_treated - matched_control) / np.sqrt(len(matched_treated))
        else:
            ate_psm = np.nan
            se_psm = np.nan

        return {
            'IPW_ATE': ate_ipw, 'IPW_SE': se_ipw,
            'PSM_ATE': ate_psm, 'PSM_SE': se_psm
        }


def bootstrap_analysis(data, outcome, treatment, confounders, n_bootstrap=1000):
    boot_results = {1: [], 2: [], 3: [], 4: []}

    # Simple progress print since this can be slow
    print(f"Bootstrapping {n_bootstrap} iterations...")

    for i in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True, random_state=i)
        est = MultinomialPropensityScore(sample, outcome, treatment, confounders)
        ates = est.ipw_ate()
        for k, v in ates.items():
            boot_results[k].append(v)

    return boot_results


def run_full_analysis():
    data = clean_data()
    paths = get_project_paths()

    # Define variables
    confounders = ['p2002', 'g2002', 'p2000', 'yob']
    treatment_names = {1: 'Civic Duty', 2: 'Hawthorne', 3: 'Self', 4: 'Neighbors'}

    estimator = MultinomialPropensityScore(data, 'voted', 'treatment_numeric', confounders)
    ps_scores = estimator.estimate_propensity_scores()

    ipw_ates = estimator.ipw_ate()
    psm_ates = estimator.psm_ate(n_neighbors=1, caliper=0.2)  # Caliper matches your Table 6 note

    # Create Table 6 Data
    table6_rows = []
    for t in [1, 2, 3, 4]:
        ipw_val = ipw_ates.get(t, np.nan)
        psm_val = psm_ates.get(t, np.nan)
        table6_rows.append({
            'Treatment Group': treatment_names[t],
            'IPW ATE': ipw_val,
            'PSM ATE': psm_val,
            'Difference (PSM-IPW)': psm_val - ipw_val
        })

    df_table6 = pd.DataFrame(table6_rows)

    # Save Table 6
    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_table6.to_latex(
        output_dir + 'table6.tex',
        float_format="%.4f",
        index=False,
        caption="Comparison of IPW and PSM Estimates",
        label="tab:ipw_psm_comparison"
    )

    treated_data = data[data['treatment_numeric'] > 0].copy()
    intensity_est = TreatmentIntensityAnalysis(treated_data, 'voted', 'high_block_intensity', confounders)
    int_results = intensity_est.analyze()

    # Create Table 7 Data
    table7_rows = [
        {
            'Method': 'IPW',
            'ATE': int_results['IPW_ATE'],
            'SE': int_results['IPW_SE'],
            '95% CI Lower': int_results['IPW_ATE'] - 1.96 * int_results['IPW_SE'],
            '95% CI Upper': int_results['IPW_ATE'] + 1.96 * int_results['IPW_SE']
        },
        {
            'Method': 'PSM',
            'ATE': int_results['PSM_ATE'],
            'SE': int_results['PSM_SE'],
            '95% CI Lower': int_results['PSM_ATE'] - 1.96 * int_results['PSM_SE'],
            '95% CI Upper': int_results['PSM_ATE'] + 1.96 * int_results['PSM_SE']
        }
    ]

    df_table7 = pd.DataFrame(table7_rows)

    # Save Table 7
    df_table7.to_latex(
        output_dir + 'table7.tex',
        float_format="%.4f",
        index=False,
        caption="Treatment Intensity Effects: IPW vs. PSM Estimates",
        label="tab:intensity_effects"
    )

    plot_dir = paths['plots']
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, t in enumerate([1, 2, 3, 4]):
        # Get score for this specific treatment column
        treat_idx = np.where(estimator.treatment_values == t)[0][0]
        current_ps = ps_scores[:, treat_idx]

        treated_mask = (data['treatment_numeric'] == t)
        control_mask = (data['treatment_numeric'] == 0)

        treated_dist = current_ps[treated_mask]
        control_dist = current_ps[control_mask]

        # Calculate shared bins
        combined = np.concatenate([treated_dist, control_dist])
        ps_min, ps_max = np.percentile(combined, 0.5), np.percentile(combined, 99.5)
        bins = np.linspace(ps_min, ps_max, 50)

        treated_counts, _ = np.histogram(treated_dist, bins=bins, density=True)
        control_counts, _ = np.histogram(control_dist, bins=bins, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ax = axes[idx]
        # Mirrored histogram plot style
        ax.bar(bin_centers, treated_counts, width=np.diff(bins)[0], align='center',
               color='royalblue', alpha=0.6, label='Treated')
        ax.bar(bin_centers, -control_counts, width=np.diff(bins)[0], align='center',
               color='salmon', alpha=0.6, label='Control')

        # Formatting
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'{treatment_names[t]} vs Control', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        # Absolute y-labels
        yticks = ax.get_yticks()
        ax.set_yticklabels([str(abs(int(y))) for y in yticks])
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')

    fig.suptitle('Positivity Check: Common Support Across All Treatments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir + 'figure6.png', dpi=600, bbox_inches='tight')
    plt.close()

    boot_data = bootstrap_analysis(data, 'voted', 'treatment_numeric', confounders, n_bootstrap=1000)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for idx, t in enumerate([1, 2, 3, 4]):
        ates = np.array(boot_data[t])
        point_est = ipw_ates[t]

        ci_95 = np.percentile(ates, [2.5, 97.5])
        ci_90 = np.percentile(ates, [5, 95])

        ax = axes[idx]

        # Histogram
        ax.hist(ates, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Vertical lines
        ax.axvline(point_est, color='black', linewidth=2.5, linestyle='--',
                   label=f'Point Est: {point_est:.4f}')
        ax.axvline(ci_95[0], color='darkgreen', linewidth=2, linestyle=':',
                   label=f'95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]')
        ax.axvline(ci_95[1], color='darkgreen', linewidth=2, linestyle=':')
        ax.axvline(ci_90[0], color='red', linewidth=1.5, linestyle=':', alpha=0.7,
                   label=f'90% CI')
        ax.axvline(ci_90[1], color='red', linewidth=1.5, linestyle=':', alpha=0.7)

        # Shade 95% region
        ax.axvspan(ci_95[0], ci_95[1], alpha=0.1, color='green')

        ax.set_title(f'{treatment_names[t]} Bootstrap Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Treatment Effect')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Bootstrap Distributions of Average Treatment Effects (1,000 iterations)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir + 'figure7.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_full_analysis()
from data_clean import clean_data
from utils import get_project_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats


def information_diffusion_analysis():
    """
    Analyze information diffusion and spillover effects
    """
    data = clean_data()

    # voted is already 0/1 from clean_data()
    # Drop any rows with missing critical data
    data = data.dropna(subset=['voted', 'cluster', 'hh_id', 'hh_size'])

    paths = get_project_paths()

    print(f"Total observations: {len(data)}")
    print(f"Voted mean: {data['voted'].mean():.4f}")
    print(f"\nTreatment distribution:")
    print(data[['treatment_control', 'treatment_neighbors', 'treatment_self',
                'treatment_hawthorne', 'treatment_civic duty']].sum())

    # ============================================
    # 1. WITHIN-HOUSEHOLD SPILLOVERS
    # ============================================
    print("\n" + "=" * 60)
    print("1. WITHIN-HOUSEHOLD SPILLOVER ANALYSIS")
    print("=" * 60)

    # Calculate treatment exposure at household level
    hh_treatment = data.groupby('hh_id').agg({
        'treatment_neighbors': 'sum',
        'treatment_self': 'sum',
        'treatment_hawthorne': 'sum',
        'treatment_civic duty': 'sum',
        'voted': 'mean',
        'hh_size': 'first'
    }).reset_index()

    # Merge back to individual level
    data = data.merge(
        hh_treatment[['hh_id', 'treatment_neighbors', 'treatment_self',
                      'treatment_hawthorne', 'treatment_civic duty']],
        on='hh_id',
        suffixes=('', '_hh_total')
    )

    # Create indicators for "others in household treated"
    data['hh_neighbors_treated'] = (data['treatment_neighbors_hh_total'] > data['treatment_neighbors']).astype(int)
    data['hh_self_treated'] = (data['treatment_self_hh_total'] > data['treatment_self']).astype(int)
    data['hh_hawthorne_treated'] = (data['treatment_hawthorne_hh_total'] > data['treatment_hawthorne']).astype(int)
    data['hh_civic_treated'] = (data['treatment_civic duty_hh_total'] > data['treatment_civic duty']).astype(int)
    data['hh_any_treated'] = ((data['hh_neighbors_treated'] + data['hh_self_treated'] +
                               data['hh_hawthorne_treated'] + data['hh_civic_treated']) > 0).astype(int)

    # Multi-person households only
    multi_hh = data[data['hh_size'] > 1].copy()

    if len(multi_hh) > 0:
        # Test spillover among control group in multi-person households
        control_multi = multi_hh[multi_hh['treatment_control'] == 1]

        if len(control_multi) > 0 and control_multi['hh_any_treated'].sum() > 0:
            spillover_effect = control_multi.groupby('hh_any_treated')['voted'].mean()
            print("\nControl group voting rate by household treatment exposure:")
            print(spillover_effect)

            # Statistical test
            no_treated = control_multi[control_multi['hh_any_treated'] == 0]['voted']
            has_treated = control_multi[control_multi['hh_any_treated'] == 1]['voted']

            if len(has_treated) > 0 and len(no_treated) > 0:
                t_stat, p_val = stats.ttest_ind(has_treated, no_treated)
                print(f"\nWithin-household spillover test:")
                print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
        else:
            print("\nInsufficient data for within-household spillover analysis")

    # ============================================
    # 2. GEOGRAPHIC SPILLOVERS (CLUSTER LEVEL)
    # ============================================
    print("\n" + "=" * 60)
    print("2. GEOGRAPHIC SPILLOVER ANALYSIS (CLUSTER LEVEL)")
    print("=" * 60)

    # Calculate treatment intensity in each cluster
    cluster_treatment = data.groupby('cluster').agg({
        'treatment_neighbors': 'mean',
        'treatment_self': 'mean',
        'treatment_hawthorne': 'mean',
        'treatment_civic duty': 'mean',
        'treatment_control': 'mean',
        'voted': 'count'
    }).reset_index()

    cluster_treatment.columns = ['cluster', 'pct_neighbors', 'pct_self',
                                 'pct_hawthorne', 'pct_civic', 'pct_control', 'cluster_size']
    cluster_treatment['treatment_intensity'] = 1 - cluster_treatment['pct_control']

    # Merge back
    data = data.merge(cluster_treatment[['cluster', 'treatment_intensity', 'cluster_size']],
                      on='cluster', how='left')

    # Test: Does treatment intensity affect control group voting?
    control_only = data[data['treatment_control'] == 1].copy()

    if len(control_only) > 0:
        # Divide into high vs low treatment intensity areas
        median_intensity = control_only['treatment_intensity'].median()
        control_only['high_intensity'] = (control_only['treatment_intensity'] > median_intensity).astype(int)

        intensity_effect = control_only.groupby('high_intensity')['voted'].mean()
        print("\nControl group voting rate by cluster treatment intensity:")
        print(f"Low intensity areas: {intensity_effect[0]:.4f}")
        print(f"High intensity areas: {intensity_effect[1]:.4f}")

        low_int = control_only[control_only['high_intensity'] == 0]['voted']
        high_int = control_only[control_only['high_intensity'] == 1]['voted']
        t_stat, p_val = stats.ttest_ind(high_int, low_int)
        print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
    else:
        print("No control group observations available")
        median_intensity = 0.5

    # ============================================
    # 3. BLOCK-LEVEL SPILLOVERS (FINER GEOGRAPHY)
    # ============================================
    print("\n" + "=" * 60)
    print("3. BLOCK-LEVEL SPILLOVER ANALYSIS")
    print("=" * 60)

    # Calculate treatment at block level
    block_treatment = data.groupby('block').agg({
        'treatment_neighbors': 'mean',
        'treatment_self': 'mean',
        'treatment_hawthorne': 'mean',
        'treatment_civic duty': 'mean',
        'treatment_control': 'mean',
        'voted': 'count'
    }).reset_index()

    block_treatment.columns = ['block', 'block_pct_neighbors', 'block_pct_self',
                               'block_pct_hawthorne', 'block_pct_civic',
                               'block_pct_control', 'block_size']
    block_treatment['block_treatment_intensity'] = 1 - block_treatment['block_pct_control']

    data = data.merge(block_treatment[['block', 'block_treatment_intensity', 'block_size']],
                      on='block', how='left')

    # ============================================
    # 4. TREATMENT EFFECT HETEROGENEITY BY INTENSITY
    # ============================================
    print("\n" + "=" * 60)
    print("4. TREATMENT EFFECTS BY CLUSTER INTENSITY")
    print("=" * 60)

    # Do treatments work better in high-saturation areas?
    data['high_cluster_intensity'] = (data['treatment_intensity'] > median_intensity).astype(int)

    # Create summary table
    results_list = []

    for treatment in ['neighbors', 'self', 'hawthorne', 'civic duty']:
        treated = data[data[f'treatment_{treatment}'] == 1]
        control = data[data['treatment_control'] == 1]

        if len(treated) > 0 and len(control) > 0:
            # Treatment effect in low intensity areas
            low_treated = treated[treated['high_cluster_intensity'] == 0]['voted'].mean()
            low_control = control[control['high_cluster_intensity'] == 0]['voted'].mean()
            low_effect = low_treated - low_control

            # Treatment effect in high intensity areas
            high_treated = treated[treated['high_cluster_intensity'] == 1]['voted'].mean()
            high_control = control[control['high_cluster_intensity'] == 1]['voted'].mean()
            high_effect = high_treated - high_control

            print(f"\n{treatment.title()} Treatment:")
            print(f"  Effect in low-intensity clusters: {low_effect:.4f}")
            print(f"  Effect in high-intensity clusters: {high_effect:.4f}")
            print(f"  Difference: {high_effect - low_effect:.4f}")

            results_list.append({
                'Treatment': treatment.title(),
                'Low Intensity Effect': low_effect,
                'High Intensity Effect': high_effect,
                'Difference': high_effect - low_effect
            })

    # Save treatment effects table
    if len(results_list) > 0:
        results_df = pd.DataFrame(results_list)
        latex_table = results_df.to_latex(
            index=False,
            float_format='%.4f',
            caption='Treatment Effects by Cluster Intensity',
            label='tab:treatment_intensity',
            escape=False
        )

        # Add resizebox
        latex_table = latex_table.replace(
            '\\begin{tabular}',
            '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}'
        )
        latex_table = latex_table.replace(
            '\\end{tabular}',
            '\\end{tabular}%\n}'
        )

        with open(paths['tables'] + 'treatment_effects_by_intensity.tex', 'w') as f:
            f.write(latex_table)

    # ============================================
    # 5. REGRESSION ANALYSIS WITH SPILLOVERS
    # ============================================
    print("\n" + "=" * 60)
    print("5. REGRESSION ANALYSIS")
    print("=" * 60)

    # Rename for regression and handle missing values
    data['treatment_civic_duty'] = data['treatment_civic duty']

    # Select variables for regression and drop missing
    reg_data = data[['voted', 'treatment_neighbors', 'treatment_self', 'treatment_hawthorne',
                     'treatment_civic_duty', 'treatment_intensity', 'yob', 'sex',
                     'g2000', 'g2002', 'p2000', 'p2002', 'p2004', 'cluster']].copy()

    reg_data = reg_data.dropna()

    print(f"Observations in regression: {len(reg_data)}")

    if len(reg_data) > 100:
        # Main regression: Include spillover effects
        formula = '''voted ~ treatment_neighbors + treatment_self + treatment_hawthorne + 
                     treatment_civic_duty + treatment_intensity + 
                     yob + sex + g2000 + g2002 + p2000 + p2002 + p2004'''

        try:
            model1 = smf.ols(formula, data=reg_data).fit(cov_type='cluster', cov_kwds={'groups': reg_data['cluster']})
            print("\nMain Model with Cluster Treatment Intensity:")
            print(model1.summary())

            # Save regression results
            with open(paths['tables'] + 'regression_spillovers.tex', 'w') as f:
                f.write(model1.summary().as_latex())
        except Exception as e:
            print(f"Error in main regression: {e}")

        # Interaction model
        formula_interact = '''voted ~ treatment_neighbors + treatment_self + treatment_hawthorne + 
                              treatment_civic_duty + treatment_intensity + 
                              treatment_neighbors:treatment_intensity +
                              yob + sex + g2000 + g2002'''

        try:
            model2 = smf.ols(formula_interact, data=reg_data).fit(cov_type='cluster',
                                                                  cov_kwds={'groups': reg_data['cluster']})
            print("\nInteraction Model (Treatment × Intensity):")
            print(model2.summary())

            latex_summary = model2.summary().as_latex()
            rename_map = {
                'yob': 'Year of Birth',
                'sex': 'Sex',
                'voted': 'Voted',
                'treatment\\_neighbors:treatment\\_intensity': 'Neighbors $\\times$ Intensity',
                'treatment\\_neighbors': 'Neighbors',
                'treatment\\_self': 'Self',
                'treatment\\_hawthorne': 'Hawthorne',
                'treatment\\_civic\\_duty': 'Civic Duty',
                'treatment\\_intensity': 'Treatment Intensity',
                'g2000': '2000 General Election',
                'g2002': '2002 General Election',
                'p2000': '2000 Primary Election',
                'p2002': '2002 Primary Election',
                'p2004': '2004 Primary Election'
            }
            for old, new in rename_map.items():
                latex_summary = latex_summary.replace(old, new)

            with open(paths['tables'] + 'regression_interactions.tex', 'w') as f:
                f.write(latex_summary)
        except Exception as e:
            print(f"Error in interaction regression: {e}")
    else:
        print("Insufficient data for regression analysis")

    # ============================================
    # 6. VISUALIZATIONS
    # ============================================

    # Plot 1: Treatment intensity distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(cluster_treatment['treatment_intensity'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Treatment Intensity (Proportion Treated in Cluster)')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Treatment Intensity Across Clusters')
    ax.axvline(median_intensity, color='red', linestyle='--', label=f'Median = {median_intensity:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths['plots'] + 'treatment_intensity_distribution.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot 2: Voting rate by treatment intensity (control group)
    if len(control_only) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Bin treatment intensity
        control_only['intensity_bin'] = pd.cut(control_only['treatment_intensity'], bins=10)
        intensity_voting = control_only.groupby('intensity_bin', observed=True)['voted'].mean()

        bin_centers = [interval.mid for interval in intensity_voting.index]
        ax.plot(bin_centers, intensity_voting.values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Treatment Intensity in Cluster')
        ax.set_ylabel('Voting Rate (Control Group Only)')
        ax.set_title('Spillover Effect: Control Group Voting by Neighborhood Treatment Intensity')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(paths['plots'] + 'spillover_by_intensity.png', dpi=600, bbox_inches='tight')
        plt.close()

    # Plot 3: Treatment effects by intensity
    if len(results_list) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        treatments = ['neighbors', 'self', 'hawthorne', 'civic duty']
        low_effects = []
        high_effects = []

        for treatment in treatments:
            treated = data[data[f'treatment_{treatment}'] == 1]
            control = data[data['treatment_control'] == 1]

            if len(treated) > 0 and len(control) > 0:
                low_treated = treated[treated['high_cluster_intensity'] == 0]['voted'].mean()
                low_control = control[control['high_cluster_intensity'] == 0]['voted'].mean()
                low_effects.append(low_treated - low_control)

                high_treated = treated[treated['high_cluster_intensity'] == 1]['voted'].mean()
                high_control = control[control['high_cluster_intensity'] == 1]['voted'].mean()
                high_effects.append(high_treated - high_control)

        x = np.arange(len(low_effects))
        width = 0.35

        ax.bar(x - width / 2, low_effects, width, label='Low Intensity Clusters', alpha=0.8)
        ax.bar(x + width / 2, high_effects, width, label='High Intensity Clusters', alpha=0.8)

        ax.set_ylabel('Treatment Effect on Voting')
        ax.set_xlabel('Treatment Type')
        ax.set_title('Treatment Effects by Cluster Treatment Intensity')
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in treatments[:len(low_effects)]])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(paths['plots'] + 'treatment_effects_by_intensity.png', dpi=600, bbox_inches='tight')
        plt.close()

    # Plot 4: Household spillovers
    if len(multi_hh) > 0:
        control_multi = multi_hh[multi_hh['treatment_control'] == 1]
        if len(control_multi) > 0 and control_multi['hh_any_treated'].sum() > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            hh_spillover = control_multi.groupby('hh_any_treated')['voted'].agg(['mean', 'count'])

            ax.bar([0, 1], hh_spillover['mean'].values, color=['coral', 'steelblue'], alpha=0.8)
            ax.set_xlabel('Other Household Member(s) Received Treatment')
            ax.set_ylabel('Voting Rate')
            ax.set_title('Within-Household Spillover Effect\n(Control Group in Multi-Person Households)')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'])

            # Add value labels and sample sizes
            for i, (rate, count) in enumerate(zip(hh_spillover['mean'].values, hh_spillover['count'].values)):
                ax.text(i, rate + 0.01, f'{rate:.3f}\n(n={int(count)})',
                        ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.savefig(paths['plots'] + 'household_spillover.png', dpi=600, bbox_inches='tight')
            plt.close()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Plots saved to:", paths['plots'])
    print("Tables saved to:", paths['tables'])
    print("=" * 60)

    return data


information_diffusion_analysis()
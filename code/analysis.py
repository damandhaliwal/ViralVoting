from data_clean import clean_data
from utils import get_project_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# print summary statistics of the data with a few variables
def summary_statistics():
    """
    Prints summary statistics of the data with a few variables.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
    """

    data = clean_data()

    # Select relevant columns for summary statistics
    summary_cols = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002', 'voted', 'treatment_control', 'treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne']
    summary_data = data[summary_cols]

    # Rename columns to be cleaner (NO underscores for LaTeX)
    summary_data = summary_data.rename(columns={
        'treatment_control': 'Control',
        'treatment_self': 'Self',
        'treatment_civic duty': 'Civic Duty',
        'treatment_neighbors': 'Neighbors',
        'treatment_hawthorne': 'Hawthorne',
        'yob': 'Year of Birth',
        'g2000': 'Voted in 2000 General Elections',
        'g2002': 'Voted in 2002 General Elections',
        'g2004': 'Voted in 2004 General Elections',
        'p2000': 'Voted in 2000 Primary Elections',
        'p2002': 'Voted in 2002 Primary Elections',
    })

    basic_stats = summary_data.describe(include='all')
    
    # Remove quartile rows (25%, 50%, 75%)
    rows_to_drop = ['25%', '50%', '75%']
    basic_stats = basic_stats.drop(index=rows_to_drop, errors='ignore')
    
    # Add counts of 0s and 1s for binary variables
    binary_cols = ['sex', 'voted', 'Control', 'Self', 'Civic Duty', 'Neighbors', 'Hawthorne',
                   'Voted in 2000 General Elections', 'Voted in 2002 General Elections', 'Voted in 2004 General Elections',
                   'Voted in 2000 Primary Elections', 'Voted in 2002 Primary Elections']
    
    # Add count_0 and count_1 rows
    count_0 = pd.Series(index=basic_stats.columns)
    count_1 = pd.Series(index=basic_stats.columns)
    
    for col in binary_cols:
        if col in summary_data.columns:
            count_0[col] = (summary_data[col] == 0).sum()
            count_1[col] = (summary_data[col] == 1).sum()
    
    # Add the new rows to the summary statistics
    basic_stats.loc['Zeros'] = count_0
    basic_stats.loc['Ones'] = count_1
    
    latex_output = basic_stats.to_latex(
        caption='Summary Statistics',
        label='tab:summary_stats',
        escape=False,
        column_format='l' + 'r' * len(basic_stats.columns),
        na_rep='-',
        float_format='%.2f'
    )

    paths = get_project_paths()
    with open(paths['tables'] + 'table1.tex', 'w') as f:
        f.write(latex_output)


def treatment_covariate_balance_table(reference_year: int = 2004):
    """Generate household-level covariate balance table by treatment group."""
    data = clean_data().copy()
    paths = get_project_paths()

    data['age_years'] = reference_year - data['yob']

    group_map = {
        'Control': 'treatment_control',
        'Civic Duty': 'treatment_civic duty',
        'Hawthorne': 'treatment_hawthorne',
        'Self': 'treatment_self',
        'Neighbors': 'treatment_neighbors',
    }

    variable_map = {
        'Household size': 'hh_size',
        'Nov 2002': 'g2002',
        'Nov 2000': 'g2000',
        'Aug 2004': 'p2004',
        'Aug 2002': 'p2002',
        'Aug 2000': 'p2000',
        'Female': 'sex',
        'Age (in years)': 'age_years',
    }

    group_frames = {
        group: data[data[indicator] == 1].copy()
        for group, indicator in group_map.items()
    }

    summary_data = pd.DataFrame(
        {
            group: [group_frames[group][column].mean() for column in variable_map.values()]
            for group in group_map
        },
        index=list(variable_map.keys()),
    )

    sample_sizes = {group: len(frame) for group, frame in group_frames.items()}

    def _format_mean(value: float) -> str:
        if pd.isna(value):
            return ''
        formatted = f'{value:.2f}'
        if abs(value) < 1:
            formatted = formatted.lstrip('0')
        return formatted

    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\caption{Relationship between Treatment Group Assignment and Covariates (Household-Level Data)}')
    lines.append('\\label{tab:household_balance}')
    lines.append('\\begin{tabular}{l' + 'c' * len(group_map) + '}')
    lines.append('\\toprule')
    lines.append(' & ' + ' & '.join(group_map.keys()) + ' \\\\')
    lines.append('\\midrule')
    lines.append(' & ' + ' & '.join(['Mean'] * len(group_map)) + ' \\\\')
    lines.append('\\midrule')

    for row_label in variable_map.keys():
        row_values = [_format_mean(summary_data.loc[row_label, group]) for group in group_map]
        lines.append(f'{row_label} & ' + ' & '.join(row_values) + ' \\\\')

    lines.append('\\midrule')
    n_values = [f"{sample_sizes[group]:,}" for group in group_map]
    lines.append('$N$ = & ' + ' & '.join(n_values) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\\\')
    lines.append('\\begin{minipage}{0.95\\textwidth}')
    lines.append('\\end{minipage}')
    lines.append('\\end{table}')
    
    latex_table = '\n'.join(lines)
    output_path = paths['tables'] + 'treatment_covariate_balance.tex'
    with open(output_path, 'w') as handle:
        handle.write(latex_table)

    print(f'Saved treatment covariate balance table to: {output_path}')


if __name__ == '__main__':
    summary_statistics()
    treatment_covariate_balance_table()


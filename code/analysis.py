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

    # Convert voted from yes/no to 1/0
    data['voted'] = data['voted'].map({'yes': 1, 'no': 0})

    # Convert treatment variables to categorical
    treatment_dummies = pd.get_dummies(data['treatment'], prefix='treatment')
    treatment_dummies = treatment_dummies.astype(int)
    data = pd.concat([data, treatment_dummies], axis=1)

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


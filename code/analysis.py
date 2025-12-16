# create summary statistics and basic correlation for data description section
# Daman Dhaliwal

# import libraries
from data_clean import clean_data
from utils import get_project_paths
import pandas as pd
import os

# print summary statistics of the data with a few variables
def summary_stats():
    data = clean_data()
    data['age'] = 2006 - data['yob']

    stats_list = []
    # for each treatment let's calculate the proportion of gender, age, previous voting history, household income
    for treatment in data['treatment'].unique():
        subset = data[data['treatment'] == treatment]

        stats = {
            'Treatment': treatment.title(),
            'N': len(subset),
            'Female': subset['sex'].mean(),
            'Avg Age': subset['age'].mean(),
            'Voted 2004 Prop': subset['p2004'].mean(),
            'Avg HH Income': subset['median_income'].mean(),
            'Outcome': subset['voted'].mean()
        }
        stats_list.append(stats)

    summary_df = pd.DataFrame(stats_list).set_index('Treatment')

    # output to latex as table 1
    path = get_project_paths()
    output_dir = path['tables']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'table1.tex'

    summary_df.to_latex(
        output_path,
        float_format="%.3f"
    )
    return

# negative correlation/guilt analysis
def guilt_analysis():
    data = clean_data()
    paths = get_project_paths()

    grouped = data.groupby(['treatment', 'p2004'])['voted'].mean().unstack()

    grouped.columns = ['No Vote 2004', 'Vote 2004']

    order = ['control', 'civic duty', 'hawthorne', 'self', 'neighbors']
    grouped = grouped.reindex(order)
    grouped.index = grouped.index.str.title()

    control_no_vote = grouped.loc['Control', 'No Vote 2004']
    control_vote = grouped.loc['Control', 'Vote 2004']

    grouped['Lift Non Voters'] = grouped['No Vote 2004'] - control_no_vote
    grouped['Lift Voters'] = grouped['Vote 2004'] - control_vote

    output_dir = paths['tables']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'table2.tex'

    grouped.to_latex(
        output_path,
        float_format="%.3f"
    )
    return

def balance_table():
    data = clean_data()
    paths = get_project_paths()

    data['Age (in years)'] = 2006 - data['yob']

    data['Household Size'] = data.groupby('hh_id')['hh_id'].transform('count')

    var_map = {
        'g2002': 'Nov 2002',
        'g2000': 'Nov 2000',
        'p2004': 'Aug 2004',
        'p2002': 'Aug 2002',
        'p2000': 'Aug 2000',
        'sex': 'Female'  # We will check if this needs 1/0 conversion
    }

    cols_to_check = ['g2002', 'g2000', 'p2002', 'p2000']
    for col in cols_to_check:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = data[col].map({'yes': 1, 'no': 0})

    data = data.rename(columns=var_map)

    covariates = [
        'Household Size',
        'Nov 2002',
        'Nov 2000',
        'Aug 2004',
        'Aug 2002',
        'Aug 2000',
        'Female',
        'Age (in years)'
    ]

    balance_df = data.groupby('treatment')[covariates].mean().T
    n_counts = data['treatment'].value_counts()

    balance_df.loc['N'] = n_counts

    cols_order = ['control', 'civic duty', 'hawthorne', 'self', 'neighbors']
    balance_df = balance_df[cols_order]

    # Capitalize Column Headers
    balance_df.columns = [c.title() for c in balance_df.columns]

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'table3.tex'

    balance_df.to_latex(
        output_path,
        float_format="%.2f"
    )
    return

if __name__ == "__main__":
    summary_stats()
    guilt_analysis()
    balance_table()

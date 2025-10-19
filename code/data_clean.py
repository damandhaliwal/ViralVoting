from utils import get_project_paths
import pandas as pd


def clean_data():
    path = get_project_paths()
    data = pd.read_csv(path['data'] + 'social.csv')

    data = data.copy()

    # Convert voted from yes/no to 1/0
    data['voted'] = data['voted'].map({'yes': 1, 'no': 0})
    data['p2004'] = data['p2004'].map({'yes': 1, 'no': 0})

    # Convert treatment variables to categorical dummies
    treatment_dummies = pd.get_dummies(data['treatment'], prefix='treatment')
    treatment_dummies = treatment_dummies.astype(int)
    data = pd.concat([data, treatment_dummies], axis=1)

    # Create the neighborhood treatment intensity variable
    cluster_variable = 'block'
    cluster_stats = cluster_creation(data, cluster_variable)

    data = data.merge(
        cluster_stats[[cluster_variable, 'treatment_intensity', 'cluster_size']],
        on=cluster_variable,
        how='left'
    )

    # Fill missing values with 0 (for blocks not in cluster_stats)
    data['treatment_intensity'] = data['treatment_intensity'].fillna(0)
    data['cluster_size'] = data['cluster_size'].fillna(0)

    median_intensity = data['treatment_intensity'].median()
    data['high_block_intensity'] = (data['treatment_intensity'] > median_intensity).astype(int)

    return data


def cluster_creation(data, variable):
    # Group by hh_id so that larger households do not skew
    hh_level = data.groupby('hh_id').agg({
        variable: 'first',
        'treatment_control': 'first'
    }).reset_index()

    cluster_stats = hh_level.groupby(variable).agg({
        'treatment_control': 'mean',
        'hh_id': 'count'}).reset_index()

    cluster_stats.columns = [variable, 'treatment_control', 'cluster_size']
    cluster_stats['treatment_intensity'] = 1 - cluster_stats['treatment_control']

    return cluster_stats
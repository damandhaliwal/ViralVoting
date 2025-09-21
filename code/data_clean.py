from utils import get_project_paths
import pandas as pd

def clean_data():
    """
    Loads and cleans the social pressure voting data.

    Returns:
        pd.DataFrame: Cleaned dataset with:
            - voted converted from yes/no to 1/0
            - treatment dummy variables created
            - ready for analysis
    """
    path = get_project_paths()
    data = pd.read_csv(path['data'] + 'social.csv')

    data = data.copy()

    # Convert voted from yes/no to 1/0
    data['voted'] = data['voted'].map({'yes': 1, 'no': 0})

    # Convert treatment variables to categorical dummies
    treatment_dummies = pd.get_dummies(data['treatment'], prefix='treatment')
    treatment_dummies = treatment_dummies.astype(int)
    data = pd.concat([data, treatment_dummies], axis=1)

    return data
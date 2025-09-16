# Replication code for ECO1465 Term Project
from analysis import summary_statistics
import utils
import pandas as pd

# load data
path = utils.get_project_paths()
data = pd.read_csv(path['data'] + 'social.csv')

# print summary statistics of key variables
summary_statistics(data)
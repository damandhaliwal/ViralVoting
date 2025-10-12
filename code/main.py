# Replication code for ECO1465 Term Project
from analysis import summary_statistics
from ols import generate_regression_table
from tree_forest import regression_tree, classification_tree, random_forest_classifier

# print summary statistics of key variables
summary_statistics()

# run all regression analyses
generate_regression_table()

# run decision tree analysis
classification_tree()

# run random_forest
random_forest_classifier()

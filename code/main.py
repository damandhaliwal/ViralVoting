# Replication code for ECO1465 Term Project
from analysis import summary_statistics
from ols import run_ols_regression, generate_regression_table
from tree_forest import regression_tree, classification_tree, random_forest_classifier

# print summary statistics of key variables - table 1
summary_statistics()

# regression table - OLS table 2
model1 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne'])
model2 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne'], control_vars = ['sex', 'yob', 'p2004'])
model3 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne', 'treatment_intensity'], control_vars = ['sex', 'yob', 'p2004'])
generate_regression_table(models = [model1, model2, model3],
                          model_names=['Basic', 'Basic + Covariates', 'Diffusion'],
                          filename = 'table2.tex')

# spatial autoregressive model - table 3
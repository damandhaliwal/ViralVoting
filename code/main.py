# Replication code for ECO1465 Term Project
from analysis import summary_statistics
from lasso import run_lasso_analysis
from ridge import run_ridge_analysis
from sar import run_sar_analysis
from ols import run_ols_regression, generate_regression_table
from ml_methods import classification_tree, random_forest_classifier, run_model_comparison

# print summary statistics of key variables - table 1
summary_statistics()

# regression table - OLS table 3
model1 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne'])
model2 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne'], control_vars = ['sex', 'yob', 'p2004'])
model3 = run_ols_regression(outcome = 'voted', treatment_vars = ['treatment_self', 'treatment_civic duty', 'treatment_neighbors', 'treatment_hawthorne', 'treatment_intensity'], control_vars = ['sex', 'yob', 'p2004'])
generate_regression_table(models = [model1, model2, model3],
                          model_names=['Basic', 'Basic + Covariates', 'Diffusion'],
                          filename = 'table3.tex')

# spatial autoregressive model - table 4
run_sar_analysis()

# lasso and ridge plots - figure 4
run_lasso_analysis()
run_ridge_analysis()

# Graphical Neural Nets Results


# Machine Learning Results
classification_tree()
random_forest_classifier()
run_model_comparison()
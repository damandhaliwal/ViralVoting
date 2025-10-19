# Replication code for ECO1465 Term Project
from analysis import summary_statistics, treatment_covariate_balance_table, create_spillover_intensity_plot
from lasso import run_lasso_analysis
from ridge import run_ridge_analysis
from sar import run_sar_analysis
from ols import run_ols_regression, generate_regression_table
from ml_methods import classification_tree, random_forest_classifier, run_model_comparison
from dag import create_dag_plots

# print summary statistics of key variables - table 1
summary_statistics()

# table 2 - covariate balance treatment matrix
treatment_covariate_balance_table()

# figure 2 - DAG
create_dag_plots()

# figure 3 - spillover by intensity
create_spillover_intensity_plot()

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

# Machine Learning Results - produces figure 5
classification_tree()

# figure 6
random_forest_classifier()

# table 5 - add include_gnn = true to run the Graphical Neural Net (will take a while to run)
run_model_comparison(include_gnn = True)
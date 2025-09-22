# Replication code for ECO1465 Term Project
from analysis import summary_statistics
from ols import generate_regression_table
from ridge import generate_ridge_table, create_ridge_coefficient_plot, create_ridge_cv_plot
from lasso import generate_lasso_table, create_lasso_coefficient_plot, create_lasso_cv_plot
import utils
import pandas as pd

# print summary statistics of key variables
summary_statistics()

# run all regression analyses
generate_regression_table()
generate_ridge_table()
generate_lasso_table()

# create Ridge plots
create_ridge_coefficient_plot()
create_ridge_cv_plot()

# create Lasso plots
create_lasso_coefficient_plot()
create_lasso_cv_plot()
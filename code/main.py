# Replication code for ECO1465 Term Project
from analysis import summary_stats, guilt_analysis, balance_table, plot_spillover
from lasso import run_lasso_analysis
from ridge import run_ridge_analysis
from sar import run_sar_analysis
from ols import ols_analysis
from ml_methods import run_model_comparison
from dag import create_dag_plots
from ipw import run_full_analysis
from learners import main
from doubly_robust import run_dr_with_interactions
from double_ml import run_double_ml_analysis

# print summary statistics of key variables - table 1
summary_stats()

# table 2 - guilt analysis
guilt_analysis()

# table 3 - balance table
balance_table()

# figure 1 - DAG
create_dag_plots()

# figure 2 - spillover by treatment
plot_spillover()

# figure 3 - spillover by intensity
create_spillover_intensity_plot()

# regression table - OLS table 4
ols_analysis()

# table 5 - SAR results
run_sar_analysis()

# figure 3a and 3b - Ridge and Lasso results
run_ridge_analysis(plot_top_n=10)
run_lasso_analysis(plot_top_n=10)

#figure 4, 5 and table 6 - ensemble results
run_model_comparison()

# table 7 and 8 - IPW
run_full_analysis()

# table 9 and 10 - Learners
main()

# table 11 and 12 - Doubly Robust Methods
run_dr_with_interactions()

# table 13 and 14 - Double ML and Causal Forests
run_double_ml_analysis()
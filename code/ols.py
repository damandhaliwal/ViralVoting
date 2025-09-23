from data_clean import clean_data
from utils import get_project_paths
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def run_ols_analysis():
    """
    Runs OLS regression analysis on the cleaned voting data.

    Returns:
        dict: Dictionary containing regression results and summary statistics
    """
    # Get cleaned data
    data = clean_data()

    # Define outcome variable
    y = data['voted']

    # Define treatment variables (exclude control group as reference)
    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne',
                      'treatment_neighbors', 'treatment_self']

    # Define control variables
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']

    # Create feature matrix with treatment and control variables
    X = data[treatment_vars + control_vars]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Run OLS regression
    model = sm.OLS(y, X).fit(cov_type = 'HC1')

    return {
        'model': model,
        'summary': model.summary(),
        'coefficients': model.params,
        'pvalues': model.pvalues,
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj
    }

def generate_regression_table():
    """
    Generates and saves a LaTeX regression table.
    """
    results = run_ols_analysis()
    path = get_project_paths()

    # Extract key results
    model = results['model']

    # Create LaTeX table
    latex_table = model.summary().as_latex()

    # Save to file
    with open(path['tables'] + 'regression_table.tex', 'w') as f:
        f.write(latex_table)

    print("Regression results saved to Output/Tables/regression_table.tex")
    print("\nRegression Summary:")
    print(model.summary())

    return results


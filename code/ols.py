from data_clean import clean_data
from utils import get_project_paths, get_clean_variable_names
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


def run_ols_regression(
        outcome='voted',
        treatment_vars=None,
        control_vars=None,
        cluster_var=None
):
    # Get cleaned data
    data = clean_data()

    y = data[outcome]

    # Default specifications if not provided
    if treatment_vars is None:
        treatment_vars = ['treatment_civic duty', 'treatment_hawthorne',
                          'treatment_neighbors', 'treatment_self']

    if control_vars is None:
        all_vars = treatment_vars
    else:
        all_vars = treatment_vars + control_vars

    X = data[all_vars].copy()

    X = sm.add_constant(X)

    # Determine covariance type
    if cluster_var:
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_var]}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    return model


def generate_regression_table(
        models,
        model_names=None,
        filename='regression_table.tex',
        title='Regression Results',
        stars=True
):
    paths = get_project_paths()

    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]

    # Get clean variable names
    name_map = get_clean_variable_names()

    # Generate summary table with summary_col
    # Don't add R-squared to info_dict since it's already included by default
    latex_table = summary_col(
        models,
        stars=stars,
        model_names=model_names,
        info_dict={
            'N': lambda x: f"{int(x.nobs):,}",
        },
        float_format='%.4f',
        regressor_order=None  # Keep original order
    ).as_latex()

    # Replace variable names with clean names
    for old_name, new_name in name_map.items():
        latex_table = latex_table.replace(old_name, new_name)

    # Remove outer table environment & caption
    latex_table = latex_table.replace('\\begin{table}', '')
    latex_table = latex_table.replace('\\end{table}', '')
    latex_table = latex_table.replace('\\caption{}', '')
    latex_table = latex_table.replace('\\label{}', '')

    # Add midrule before R-squared (only once)
    if 'R-squared' in latex_table:
        latex_table = latex_table.replace('R-squared', '\\midrule\nR-squared', 1)

    # Save to file
    output_path = paths['tables'] + filename
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"Regression table saved to {output_path}")

    return latex_table
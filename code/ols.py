# ols method with interaction terms
# Daman Dhaliwal
import os

# import libraries
from data_clean import clean_data
from utils import get_project_paths, get_clean_variable_names
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


def run_ols_regression(
        data = None,
        outcome='voted',
        treatment_vars=None,
        control_vars=None,
        cluster_var=None
):
    if data is None:
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
        filename='table4.tex',
        title='Regression Results',
        stars=True
):
    paths = get_project_paths()

    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]

    # Get clean variable names
    name_map = get_clean_variable_names()

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
    output_dir = paths['tables']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + filename
    with open(output_path, 'w') as f:
        f.write(latex_table)

    return latex_table


def ols_analysis():
    df = clean_data()

    outcome = 'voted'
    treatments = ['treatment_civic duty', 'treatment_hawthorne', 'treatment_neighbors', 'treatment_self']
    controls = ['sex', 'yob', 'p2004']

    cluster_level = 'hh_id'

    interaction_vars = []
    for treat in treatments:
        inter_name = f'{treat}_x_intensity'
        df[inter_name] = df[treat] * df['treatment_intensity']
        interaction_vars.append(inter_name)

    # Model 1: Baseline + Controls (Eq 2)
    model_baseline = run_ols_regression(
        data=df,
        outcome=outcome,
        treatment_vars=treatments,
        control_vars=controls,
        cluster_var=cluster_level
    )

    # Model 2: Diffusion (Eq 3)
    model_diffusion = run_ols_regression(
        data=df,
        outcome=outcome,
        treatment_vars=treatments + ['treatment_intensity'],
        control_vars=controls,
        cluster_var=cluster_level
    )

    # Model 3: Interaction (The "Substitutability" Test)
    model_interaction = run_ols_regression(
        data=df,
        outcome=outcome,
        treatment_vars=treatments + ['treatment_intensity'] + interaction_vars,
        control_vars=controls,
        cluster_var=cluster_level
    )

    dynamic_filename = f'table4_{cluster_level}.tex'

    generate_regression_table(
        models=[model_baseline, model_diffusion, model_interaction],
        model_names=['Baseline', 'Diffusion', 'Interaction'],
        filename=dynamic_filename,
        title=f'OLS Estimates (Clustered by {cluster_level})'
    )

    print(f"Saved results to: {dynamic_filename}")

    return

if __name__ == "__main__":
    ols_analysis()
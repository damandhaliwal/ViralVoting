# Spatial Autoregressive Model
# Daman Dhaliwal
import os

# import libraries
import numpy as np
import pandas as pd
from data_clean import clean_data
from utils import get_project_paths, get_clean_variable_names
import pgeocode
from libpysal.weights import KNN
from spreg import GM_Lag


def run_sar_analysis(num_neighbors=8):
    data = clean_data()

    if 'zip' not in data.columns:
        np.random.seed(42)
        data['zip'] = np.random.randint(48000, 49900, size=len(data))

    data['zip_clean'] = data['zip'].astype(str).str.zfill(5)
    unique_zips = data['zip_clean'].unique()

    geocoder = pgeocode.Nominatim('us')
    coordinates_list = []

    for zip_code in unique_zips:
        result = geocoder.query_postal_code(zip_code)
        if pd.notna(result['latitude']) and pd.notna(result['longitude']):
            coordinates_list.append({
                'zip_code': zip_code,
                'latitude': result['latitude'],
                'longitude': result['longitude']
            })

    coordinates_data = pd.DataFrame(coordinates_list)

    if len(coordinates_data) == 0:
        raise ValueError("No valid coordinates obtained from geocoding")

    data = data.merge(coordinates_data, left_on='zip_clean', right_on='zip_code', how='inner')

    coords = data[['latitude', 'longitude']].values
    spatial_weights = KNN(coords, k=num_neighbors)
    spatial_weights.transform = 'r'

    base_treatments = [
        'treatment_civic duty',
        'treatment_hawthorne',
        'treatment_self',
        'treatment_neighbors'
    ]

    interaction_vars = []
    for treat in base_treatments:
        inter_name = f'{treat}_x_intensity'
        data[inter_name] = data[treat] * data['treatment_intensity']
        interaction_vars.append(inter_name)

    treatment_vars = base_treatments + ['treatment_intensity'] + interaction_vars

    control_vars = ['sex', 'yob', 'p2004']

    all_features = treatment_vars + control_vars
    available_features = [f for f in all_features if f in data.columns]

    name_mapping = get_clean_variable_names()
    if 'treatment_intensity' not in name_mapping:
        name_mapping['treatment_intensity'] = 'Neighborhood Intensity'

    for treat in base_treatments:
        clean_base = name_mapping.get(treat, treat)
        inter_key = f'{treat}_x_intensity'
        name_mapping[inter_key] = f'{clean_base} x Intensity'

    clean_names = [name_mapping.get(var, var) for var in available_features]

    X = data[available_features].values
    y = data['voted'].astype(float).values

    model = GM_Lag(y, X, w=spatial_weights, name_y='Voted', name_x=clean_names)

    rho = float(model.rho[0])

    rho_variance = model.vm[-1, -1]
    rho_stderr = np.sqrt(rho_variance)

    from scipy.stats import norm
    if rho_stderr > 0:
        z_score = rho / rho_stderr
        rho_p = 2 * (1 - norm.cdf(abs(z_score)))
    else:
        rho_p = np.nan

    betas = model.betas.flatten()
    std_errors = model.std_err.flatten()

    z_statistics = model.z_stat if hasattr(model, 'z_stat') else []

    results_list = []

    results_list.append({
        'Variable': 'Constant',
        'Coefficient': betas[0],
        'Std. Error': std_errors[0],
        'p-value': z_statistics[0][1] if len(z_statistics) > 0 else np.nan
    })

    for i, name in enumerate(clean_names):
        idx = i + 1
        results_list.append({
            'Variable': name,
            'Coefficient': betas[idx],
            'Std. Error': std_errors[idx],
            'p-value': z_statistics[idx][1] if idx < len(z_statistics) else np.nan
        })

    results_list.append({
        'Variable': 'Spatial Lag ($\\rho$)',
        'Coefficient': rho,
        'Std. Error': rho_stderr,
        'p-value': rho_p
    })

    results = pd.DataFrame(results_list)

    _generate_sar_table(results, model.n, num_neighbors)

    return model, results


def _generate_sar_table(results, n_obs, num_neighbors):
    paths = get_project_paths()

    def format_with_stars(coef, pval):
        if pd.isna(pval):
            return f'{coef:.4f}'
        elif pval < 0.01:
            return f'{coef:.4f}***'
        elif pval < 0.05:
            return f'{coef:.4f}**'
        elif pval < 0.1:
            return f'{coef:.4f}*'
        return f'{coef:.4f}'

    lines = []

    lines.append('\\begin{threeparttable}')

    lines.append('\\caption{Spatial Autoregressive Model Results}')
    lines.append('\\label{tab:sar_results}')

    lines.append('\\begin{tabular}{l r}')
    lines.append('\\hline')
    lines.append('Variable & \\multicolumn{1}{c}{Coefficient} \\\\')
    lines.append('\\hline')

    for _, row in results.iterrows():
        var_name = row['Variable']
        coef_str = format_with_stars(row['Coefficient'], row['p-value'])

        if pd.notna(row['Std. Error']):
            stderr_str = f"({row['Std. Error']:.4f})"
        else:
            stderr_str = ""

        lines.append(f"{var_name} & {coef_str} \\\\")
        lines.append(f" & {stderr_str} \\\\")

    lines.append('\\midrule')
    lines.append(f'Observations & {int(n_obs):,} \\\\')
    lines.append(f'Neighbors (k) & {num_neighbors} \\\\')
    lines.append('\\hline')
    lines.append('\\end{tabular}')

    lines.append('\\begin{tablenotes}')
    lines.append('\\small')
    lines.append('\\item Standard errors in parentheses.')
    lines.append('\\item * p<.1, ** p<.05, *** p<.01')
    lines.append('\\end{tablenotes}')

    lines.append('\\end{threeparttable}')

    latex_output = '\n'.join(lines)

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + 'table5.tex'

    with open(output_path, 'w') as f:
        f.write(latex_output)

    return latex_output


if __name__ == "__main__":
    run_sar_analysis()
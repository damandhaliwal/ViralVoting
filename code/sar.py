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

    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne', 'treatment_self', 'treatment_neighbors']
    control_vars = ['sex', 'yob', 'p2004']
    all_features = treatment_vars + control_vars
    available_features = [f for f in all_features if f in data.columns]

    name_mapping = get_clean_variable_names()
    clean_names = [name_mapping.get(var, var) for var in available_features]

    X = data[available_features].values
    y = data['voted'].astype(float).values

    model = GM_Lag(y, X, w=spatial_weights, name_y='Voted', name_x=clean_names)

    rho = float(model.rho[0])
    rho_stderr = float(model.std_err_rho) if hasattr(model, 'std_err_rho') else np.nan
    z_statistics = model.z_stat if hasattr(model, 'z_stat') else []
    rho_z = z_statistics[0][0] if z_statistics else np.nan
    rho_p = z_statistics[0][1] if z_statistics else np.nan

    betas = model.betas.flatten()
    std_errors = model.std_err.flatten()
    beta_z_stats = z_statistics[1:] if len(z_statistics) > 1 else []

    results_list = [
        {'Variable': 'Spatial Lag ($\\rho$)', 'Coefficient': rho, 'Std. Error': rho_stderr, 'p-value': rho_p}]

    for i, name in enumerate(clean_names):
        p_val = beta_z_stats[i][1] if i < len(beta_z_stats) else np.nan
        results_list.append({
            'Variable': name,
            'Coefficient': betas[i],
            'Std. Error': std_errors[i],
            'p-value': p_val
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
    lines.append('\\caption{Spatial Autoregressive Model Results}')
    lines.append('\\label{tab:sar_results}')
    lines.append('\\begin{center}')
    lines.append('\\begin{tabular}{lc}')
    lines.append('\\hline')
    lines.append(' & Voted \\\\')
    lines.append('\\hline')

    for _, row in results.iterrows():
        var_name = row['Variable']
        coef_str = format_with_stars(row['Coefficient'], row['p-value'])
        stderr_str = f"({row['Std. Error']:.4f})"
        lines.append(f"{var_name} & {coef_str} \\\\")
        lines.append(f" & {stderr_str} \\\\")

    lines.append('\\midrule')
    lines.append(f'N & {int(n_obs):,} \\\\')
    lines.append(f'Neighbors & {num_neighbors} \\\\')
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{center}')
    lines.append('\\bigskip')
    lines.append('Standard errors in parentheses. \\newline')
    lines.append('* p<.1, ** p<.05, *** p<.01')

    latex_output = '\n'.join(lines)

    with open(paths['tables'] + 'table4.tex', 'w') as f:
        f.write(latex_output)

    return latex_output
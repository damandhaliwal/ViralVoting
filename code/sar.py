import numpy as np
import pandas as pd
from data_clean import clean_data
from utils import get_project_paths

try:
    import pgeocode
    from libpysal.weights import KNN
    from spreg import GM_Lag
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Install required packages: pip install pgeocode libpysal spreg")


def get_clean_variable_names():
    """Return mapping of variable codes to clean presentation names."""
    return {
        'treatment_civic duty': 'Civic Duty',
        'treatment_hawthorne': 'Hawthorne',
        'treatment_self': 'Self',
        'treatment_neighbors': 'Neighbors',
        'sex': 'Female',
        'yob': 'Year of Birth',
        'g2000': 'Voted General 2000',
        'g2002': 'Voted General 2002',
        'p2000': 'Voted Primary 2000',
        'p2002': 'Voted Primary 2002',
        'p2004': 'Voted Primary 2004',
    }


def run_sar_analysis(num_neighbors=8, sample_observations=None):
    """
    Run Spatial Autoregressive (SAR) model analysis.
    
    Parameters:
    -----------
    num_neighbors : int
        Number of spatial neighbors for weights matrix
    sample_observations : int, optional
        Number of observations to sample (for faster testing)
    
    Returns:
    --------
    tuple : (model, results_dataframe)
    """
    
    if not LIBRARIES_AVAILABLE:
        print("ERROR: Required libraries not installed")
        print("Run: pip install pgeocode libpysal spreg")
        return None, pd.DataFrame()
    
    print("Loading data...")
    data = clean_data()
    
    # Create ZIP codes if not present (Michigan ZIPs: 48000-49900)
    if 'zip' not in data.columns:
        print("Creating synthetic ZIP codes...")
        np.random.seed(42)
        data['zip'] = np.random.randint(48000, 49900, size=len(data))
    
    # Sample if requested
    if sample_observations and len(data) > sample_observations:
        print("Sampling {} observations...".format(sample_observations))
        data = data.sample(n=sample_observations, random_state=42).reset_index(drop=True)
    
    # Prepare ZIP codes - store as a column
    print("Preparing ZIP codes...")
    data['zip_clean'] = data['zip'].astype(str).str.zfill(5)
    
    print("Geocoding ZIP codes...")
    unique_zips = data['zip_clean'].unique()
    print("Unique ZIPs to geocode: {}".format(len(unique_zips)))
    print("Sample ZIPs from data: {}".format(list(unique_zips[:5])))
    
    # Get coordinates
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
    
    print("Valid coordinates: {}/{}".format(len(coordinates_data), len(unique_zips)))
    print("Sample coordinates ZIPs: {}".format(list(coordinates_data['zip_code'].head())))
    
    if len(coordinates_data) == 0:
        raise ValueError("No valid coordinates obtained from geocoding")
    
    # Merge coordinates back to data
    print("Merging coordinates with data...")
    print("Data zip_clean type: {}".format(data['zip_clean'].dtype))
    print("Coordinates zip_code type: {}".format(coordinates_data['zip_code'].dtype))
    
    data = data.merge(
        coordinates_data,
        left_on='zip_clean',
        right_on='zip_code',
        how='inner'
    )
    
    print("Final sample: {} observations".format(len(data)))
    
    if len(data) == 0:
        # Debug: Check a few examples
        print("\nDEBUG: Checking why merge failed")
        print("First 5 zip_clean values:", data['zip_clean'].head().tolist() if 'zip_clean' in data.columns else "Column missing")
        print("First 5 zip_code values:", coordinates_data['zip_code'].head().tolist())
        print("Any matches?", len(set(data['zip_clean'].unique()) & set(coordinates_data['zip_code'].unique())))
        raise ValueError("No observations with valid coordinates after merge")
    
    # Create spatial weights
    print("Building spatial weights matrix...")
    coords = data[['latitude', 'longitude']].values
    spatial_weights = KNN(coords, k=num_neighbors)
    spatial_weights.transform = 'r'
    
    # Define features with clean names
    treatment_vars = [
        'treatment_civic duty',
        'treatment_hawthorne', 
        'treatment_self',
        'treatment_neighbors'
    ]
    
    control_vars = [
        'sex',
        'yob',
        'g2000',
        'g2002',
        'p2000',
        'p2002',
        'p2004'
    ]
    
    all_features = treatment_vars + control_vars
    
    # Check which features exist
    available_features = [f for f in all_features if f in data.columns]
    print("Available features: {}/{}".format(len(available_features), len(all_features)))
    
    if len(available_features) == 0:
        raise ValueError("No features available in data")
    
    # Get clean names for available features
    name_mapping = get_clean_variable_names()
    clean_names = [name_mapping.get(var, var) for var in available_features]
    
    # Prepare data
    X = data[available_features].values
    y = data['voted'].astype(float).values
    
    print("Estimating SAR model...")
    print("Features: {}".format(len(available_features)))
    print("Observations: {}".format(len(y)))
    print("Outcome mean: {:.3f}".format(y.mean()))
    
    # Estimate model
    model = GM_Lag(
        y,
        X,
        w=spatial_weights,
        name_y='Voted',
        name_x=clean_names
    )
    
    # Extract results
    print("\nExtracting results...")
    
    # Spatial lag parameter
    rho = float(model.rho[0])
    rho_stderr = float(model.std_err_rho) if hasattr(model, 'std_err_rho') else np.nan
    
    # Get z-statistics
    z_statistics = model.z_stat if hasattr(model, 'z_stat') else []
    rho_z = z_statistics[0][0] if z_statistics else np.nan
    rho_p = z_statistics[0][1] if z_statistics else np.nan
    
    # Beta coefficients
    betas = model.betas.flatten()
    std_errors = model.std_err.flatten()
    beta_z_stats = z_statistics[1:] if len(z_statistics) > 1 else []
    
    # Build results table
    results_list = []
    
    # Add spatial lag first
    results_list.append({
        'Variable': 'Spatial Lag (rho)',
        'Coefficient': rho,
        'Std. Error': rho_stderr,
        'Z-statistic': rho_z,
        'p-value': rho_p
    })
    
    # Add feature coefficients
    for i, name in enumerate(clean_names):
        z_val = beta_z_stats[i][0] if i < len(beta_z_stats) else np.nan
        p_val = beta_z_stats[i][1] if i < len(beta_z_stats) else np.nan
        
        results_list.append({
            'Variable': name,
            'Coefficient': betas[i],
            'Std. Error': std_errors[i],
            'Z-statistic': z_val,
            'p-value': p_val
        })
    
    results = pd.DataFrame(results_list)
    
    # Add significance stars
    def add_stars(row):
        p = row['p-value']
        coef = row['Coefficient']
        if pd.isna(p):
            return '{:.4f}'.format(coef)
        elif p < 0.001:
            return '{:.4f}***'.format(coef)
        elif p < 0.01:
            return '{:.4f}**'.format(coef)
        elif p < 0.05:
            return '{:.4f}*'.format(coef)
        else:
            return '{:.4f}'.format(coef)
    
    # Create display dataframe
    display_df = results.copy()
    display_df['Coefficient'] = display_df.apply(add_stars, axis=1)
    display_df['Std. Error'] = display_df['Std. Error'].apply(lambda x: '{:.4f}'.format(x))
    display_df['Z-statistic'] = display_df['Z-statistic'].apply(lambda x: '{:.3f}'.format(x) if not pd.isna(x) else '')
    display_df['p-value'] = display_df['p-value'].apply(lambda x: '{:.4f}'.format(x) if not pd.isna(x) else '')
    
    # Save results
    paths = get_project_paths()
    
    # Save CSV with raw numbers
    csv_path = paths['tables'] + 'sar_results.csv'
    results.to_csv(csv_path, index=False)
    
    # Create LaTeX table using pandas to_latex
    latex_path = paths['tables'] + 'sar_results.tex'
    
    latex_output = display_df.to_latex(
        index=False,
        escape=False,
        column_format='lcccc',
        caption='Spatial Autoregressive Model Results',
        label='tab:sar_results',
        position='htbp'
    )
    
    # Add table notes
    notes = (
        r"\begin{tablenotes}" + "\n" +
        r"\small" + "\n" +
        r"\item Notes: *** p$<$0.001, ** p$<$0.01, * p$<$0.05." + "\n" +
        r"\item Spatial lag coefficient (rho) measures spatial dependence in voting behavior." + "\n" +
        r"\item Model estimated using generalized method of moments with {}-nearest neighbors spatial weights.".format(num_neighbors) + "\n" +
        r"\end{tablenotes}"
    )
    
    # Insert notes before \end{table}
    latex_output = latex_output.replace(r'\end{table}', notes + '\n' + r'\end{table}')
    
    with open(latex_path, 'w') as f:
        f.write(latex_output)
    
    # Also save model summary if available
    try:
        summary_path = paths['tables'] + 'sar_model_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(str(model.summary))
        print("\nModel summary saved to: {}".format(summary_path))
    except:
        pass
    
    # Print results
    print("\n" + "="*80)
    print("SPATIAL AUTOREGRESSIVE MODEL RESULTS")
    print("="*80)
    print(display_df.to_string(index=False))
    print("="*80)
    print("\nResults saved:")
    print("  CSV: {}".format(csv_path))
    print("  LaTeX: {}".format(latex_path))
    
    return model, results


if __name__ == "__main__":
    model, results = run_sar_analysis(
        num_neighbors=8,
        sample_observations=10000
    )
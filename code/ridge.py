from data_clean import clean_data
from utils import get_project_paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def run_ridge_analysis(alpha=1.0, cv_alpha=True):
    """
    Runs Ridge regression analysis on the cleaned voting data.

    Args:
        alpha (float): Regularization parameter. If cv_alpha=True, this is ignored.
        cv_alpha (bool): If True, uses cross-validation to select optimal alpha.

    Returns:
        dict: Dictionary containing Ridge regression results and summary statistics
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

    # Standardize features for Ridge regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use cross-validation to find optimal alpha if requested
    if cv_alpha:
        alphas = np.logspace(-4, 2, 50)
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(X_scaled, y)
        optimal_alpha = ridge_cv.alpha_
        model = ridge_cv
    else:
        optimal_alpha = alpha
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    return {
        'model': model,
        'optimal_alpha': optimal_alpha,
        'coefficients': model.coef_,
        'feature_names': treatment_vars + control_vars,
        'scaler': scaler,
        'r2_score': r2,
        'mse': mse,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def generate_ridge_table():
    """
    Generates and saves a Ridge regression results table.
    """
    results = run_ridge_analysis()
    path = get_project_paths()

    # Create results DataFrame
    coef_df = pd.DataFrame({
        'Variable': results['feature_names'],
        'Coefficient': results['coefficients']
    })

    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_df = coef_df.drop('Abs_Coefficient', axis=1)

    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Optimal Alpha', 'R² Score', 'MSE', 'CV Score (Mean)', 'CV Score (Std)'],
        'Value': [results['optimal_alpha'], results['r2_score'], results['mse'],
                 results['cv_mean'], results['cv_std']]
    })

    # Generate LaTeX tables
    coef_latex = coef_df.to_latex(
        caption='Ridge Regression Coefficients',
        label='tab:ridge_coefficients',
        escape=False,
        index=False,
        column_format='lr',
        float_format='%.4f'
    )

    summary_latex = summary_stats.to_latex(
        caption='Ridge Regression Summary Statistics',
        label='tab:ridge_summary',
        escape=False,
        index=False,
        column_format='lr',
        float_format='%.4f'
    )

    # Combine tables
    combined_latex = coef_latex + '\n\n' + summary_latex

    # Save to file
    with open(path['tables'] + 'ridge_table.tex', 'w') as f:
        f.write(combined_latex)

    print("Ridge regression results saved to Output/Tables/ridge_table.tex")
    print(f"\nRidge Regression Summary:")
    print(f"Optimal Alpha: {results['optimal_alpha']:.4f}")
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"Cross-Validation Score: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
    print(f"\nTop 5 Coefficients by Absolute Value:")
    print(coef_df.head())

    return results

def create_ridge_coefficient_plot():
    """
    Creates a Ridge coefficient plot showing how coefficients change with lambda (alpha).
    Saves the plot to Output/Plots/ridge_coefficients.png
    """
    # Get cleaned data
    data = clean_data()
    path = get_project_paths()

    # Define outcome variable
    y = data['voted']

    # Define treatment variables (exclude control group as reference)
    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne',
                      'treatment_neighbors', 'treatment_self']

    # Define control variables
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']

    # Create feature matrix with treatment and control variables
    X = data[treatment_vars + control_vars]
    feature_names = treatment_vars + control_vars

    # Standardize features for Ridge regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define range of alpha values (lambda in regularization)
    alphas = np.logspace(-4, 3, 100)

    # Store coefficients for each alpha
    coefficients = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        coefficients.append(ridge.coef_)

    coefficients = np.array(coefficients)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot coefficient paths
    for i, feature in enumerate(feature_names):
        plt.plot(alphas, coefficients[:, i], label=feature, linewidth=2)

    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=12)
    plt.ylabel('Coefficient Values', fontsize=12)
    plt.title('Ridge Regression: Coefficient Paths vs Lambda', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(path['plots'] + 'ridge_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Ridge coefficient plot saved to Output/Plots/ridge_coefficients.png")

def create_ridge_cv_plot():
    """
    Creates a cross-validation plot showing CV MSE vs lambda with error bars.
    Saves the plot to Output/Plots/ridge_cv_mse.png
    """
    # Get cleaned data
    data = clean_data()
    path = get_project_paths()

    # Define outcome variable
    y = data['voted']

    # Define treatment variables (exclude control group as reference)
    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne',
                      'treatment_neighbors', 'treatment_self']

    # Define control variables
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']

    # Create feature matrix with treatment and control variables
    X = data[treatment_vars + control_vars]

    # Standardize features for Ridge regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define range of alpha values (lambda in regularization)
    alphas = np.logspace(-4, 3, 50)

    # Perform cross-validation for each alpha
    cv_mse_means = []
    cv_mse_stds = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        # Use negative MSE because cross_val_score maximizes the score
        cv_scores = cross_val_score(ridge, X_scaled, y, cv=5,
                                  scoring='neg_mean_squared_error')
        # Convert back to positive MSE
        cv_mse = -cv_scores
        cv_mse_means.append(cv_mse.mean())
        cv_mse_stds.append(cv_mse.std())

    cv_mse_means = np.array(cv_mse_means)
    cv_mse_stds = np.array(cv_mse_stds)

    # Find optimal alpha (minimum CV MSE)
    optimal_idx = np.argmin(cv_mse_means)
    optimal_alpha = alphas[optimal_idx]
    optimal_mse = cv_mse_means[optimal_idx]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot CV MSE with error bars
    plt.errorbar(alphas, cv_mse_means, yerr=cv_mse_stds,
                fmt='o-', linewidth=2, markersize=4, capsize=3)

    # Mark the optimal alpha
    plt.axvline(x=optimal_alpha, color='red', linestyle='--',
               label=f'Optimal λ = {optimal_alpha:.4f}')
    plt.plot(optimal_alpha, optimal_mse, 'ro', markersize=8,
            label=f'Min CV MSE = {optimal_mse:.4f}')

    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=12)
    plt.ylabel('Cross-Validation MSE', fontsize=12)
    plt.title('Ridge Regression: Cross-Validation MSE vs Lambda', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(path['plots'] + 'ridge_cv_mse.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Ridge CV plot saved to Output/Plots/ridge_cv_mse.png")
    print(f"Optimal lambda: {optimal_alpha:.4f}, CV MSE: {optimal_mse:.4f}")


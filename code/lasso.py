from data_clean import clean_data
from utils import get_project_paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def run_lasso_analysis(alpha=1.0, cv_alpha=True):
    """
    Runs Lasso regression analysis on the cleaned voting data.

    Args:
        alpha (float): Regularization parameter. If cv_alpha=True, this is ignored.
        cv_alpha (bool): If True, uses cross-validation to select optimal alpha.

    Returns:
        dict: Dictionary containing Lasso regression results and summary statistics
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

    # Standardize features for Lasso regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use cross-validation to find optimal alpha if requested
    if cv_alpha:
        alphas = np.logspace(-4, 2, 50)
        lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
        lasso_cv.fit(X_scaled, y)
        optimal_alpha = lasso_cv.alpha_
        model = lasso_cv
    else:
        optimal_alpha = alpha
        model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        model.fit(X_scaled, y)

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    # Count selected features (non-zero coefficients)
    selected_features = np.sum(np.abs(model.coef_) > 1e-6)

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
        'cv_std': cv_scores.std(),
        'selected_features': selected_features
    }

def generate_lasso_table():
    """
    Generates and saves a Lasso regression results table.
    """
    results = run_lasso_analysis()
    path = get_project_paths()

    # Create results DataFrame
    coef_df = pd.DataFrame({
        'Variable': results['feature_names'],
        'Coefficient': results['coefficients']
    })

    # Filter out zero coefficients and sort by absolute coefficient value
    coef_df = coef_df[np.abs(coef_df['Coefficient']) > 1e-6]
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_df = coef_df.drop('Abs_Coefficient', axis=1)

    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Optimal Alpha', 'R² Score', 'MSE', 'CV Score (Mean)', 'CV Score (Std)', 'Selected Features'],
        'Value': [results['optimal_alpha'], results['r2_score'], results['mse'],
                 results['cv_mean'], results['cv_std'], results['selected_features']]
    })

    # Generate LaTeX tables
    coef_latex = coef_df.to_latex(
        caption='Lasso Regression Coefficients (Non-Zero Only)',
        label='tab:lasso_coefficients',
        escape=False,
        index=False,
        column_format='lr',
        float_format='%.4f'
    )

    summary_latex = summary_stats.to_latex(
        caption='Lasso Regression Summary Statistics',
        label='tab:lasso_summary',
        escape=False,
        index=False,
        column_format='lr',
        float_format='%.4f'
    )

    # Combine tables
    combined_latex = coef_latex + '\n\n' + summary_latex

    # Save to file
    with open(path['tables'] + 'lasso_table.tex', 'w') as f:
        f.write(combined_latex)

    print("Lasso regression results saved to Output/Tables/lasso_table.tex")
    print(f"\nLasso Regression Summary:")
    print(f"Optimal Alpha: {results['optimal_alpha']:.4f}")
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"Cross-Validation Score: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
    print(f"Selected Features: {results['selected_features']} out of {len(results['feature_names'])}")
    print(f"\nSelected Variables with Non-Zero Coefficients:")
    if len(coef_df) > 0:
        print(coef_df)
    else:
        print("No variables selected (all coefficients are zero)")

    return results

def create_lasso_coefficient_plot():
    """
    Creates a Lasso coefficient plot showing how coefficients change with lambda (alpha).
    Saves the plot to Output/Plots/lasso_coefficients.png
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

    # Standardize features for Lasso regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define range of alpha values (lambda in regularization)
    alphas = np.logspace(-4, 1, 100)

    # Store coefficients for each alpha
    coefficients = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        coefficients.append(lasso.coef_)

    coefficients = np.array(coefficients)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot coefficient paths
    for i, feature in enumerate(feature_names):
        plt.plot(alphas, coefficients[:, i], label=feature, linewidth=2)

    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=12)
    plt.ylabel('Coefficient Values', fontsize=12)
    plt.title('Lasso Regression: Coefficient Paths vs Lambda', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(path['plots'] + 'lasso_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Lasso coefficient plot saved to Output/Plots/lasso_coefficients.png")

def create_lasso_cv_plot():
    """
    Creates a cross-validation plot showing CV MSE vs lambda with error bars.
    Saves the plot to Output/Plots/lasso_cv_mse.png
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

    # Standardize features for Lasso regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define range of alpha values (lambda in regularization)
    alphas = np.logspace(-4, 1, 50)

    # Perform cross-validation for each alpha
    cv_mse_means = []
    cv_mse_stds = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        # Use negative MSE because cross_val_score maximizes the score
        cv_scores = cross_val_score(lasso, X_scaled, y, cv=5,
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
    plt.title('Lasso Regression: Cross-Validation MSE vs Lambda', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(path['plots'] + 'lasso_cv_mse.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Lasso CV plot saved to Output/Plots/lasso_cv_mse.png")
    print(f"Optimal lambda: {optimal_alpha:.4f}, CV MSE: {optimal_mse:.4f}")


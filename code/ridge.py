from data_clean import clean_data
from utils import get_project_paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, log_loss, confusion_matrix)
from sklearn.model_selection import train_test_split

def run_ridge_analysis():
    """
    Runs Ridge (L2-penalized logistic) regression analysis on voting experiment data.
    
    Returns:
        dict: Comprehensive results including model, metrics, and interpretations
    """
    # Load and prepare data
    data = clean_data()
    paths = get_project_paths()
    
    # Define variables
    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne',
                      'treatment_neighbors', 'treatment_self']
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'g2004', 'p2000', 'p2002']
    feature_names = treatment_vars + control_vars
    
    # Create feature matrix and outcome
    X = np.array(data[feature_names])
    y = np.array(data['voted'])
    
    # Train-test split (80-20) with stratification to maintain voting rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Manual standardization (fit on training data only to prevent leakage)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero
    
    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std  # Use training parameters
    
    # Set up regularization parameters (100 values from 10^-4 to 10^3)
    lambdas = np.logspace(-4, 3, 100)
    Cs = 1 / lambdas  # LogisticRegression uses C = 1/lambda
    
    # Fit Ridge logistic regression with 5-fold CV to find optimal C
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        penalty='l2',
        solver='liblinear',
        scoring='neg_log_loss',  # Better for probability calibration
        random_state=42,
        max_iter=2000,
        refit=True  # Refit on entire training set with best C
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_logloss = log_loss(y_test, y_test_proba)
    
    # Training metrics for comparison (check overfitting)
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Extract coefficients and optimal regularization
    coefficients = model.coef_[0]
    optimal_C = model.C_[0]
    optimal_lambda = 1 / optimal_C
    
    # Get CV scores for the optimal C
    cv_scores = model.scores_[1].mean(axis=0)
    best_cv_score = cv_scores[list(model.Cs_).index(optimal_C)]
    
    # Calculate marginal effects for interpretation
    # (converts log-odds to probability changes)
    test_proba_mean = y_test_proba.mean()
    marginal_effects = coefficients * test_proba_mean * (1 - test_proba_mean)
    
    # Create results dictionary
    results = {
        # Model information
        'model': model,
        'optimal_C': optimal_C,
        'optimal_lambda': optimal_lambda,
        
        # Coefficients
        'coefficients': coefficients,
        'marginal_effects': marginal_effects,
        'feature_names': feature_names,
        
        # Test set metrics (unbiased estimates)
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_logloss': test_logloss,
        
        # Training metrics (for comparison)
        'train_accuracy': train_accuracy,
        'train_auc': train_auc,
        
        # Cross-validation score
        'cv_best_score': best_cv_score,
        
        # Data info
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_vote_rate': y_train.mean(),
        'test_vote_rate': y_test.mean(),
        
        # Standardization parameters (for new data)
        'train_mean': train_mean,
        'train_std': train_std
    }
    
    # Print summary
    print_results_summary(results)
    create_coefficient_and_error_plots(results, paths)
    
    return results


def print_results_summary(results):
    """Print a clean summary of the ridge regression results."""
    print("\n" + "="*60)
    print("RIDGE REGRESSION RESULTS (L2 Penalized Logistic)")
    print("="*60)
    
    print(f"\nData Split:")
    print(f"  Training samples: {results['n_train']:,}")
    print(f"  Test samples: {results['n_test']:,}")
    print(f"  Baseline vote rate: {results['test_vote_rate']:.1%}")
    
    print(f"\nOptimal Regularization:")
    print(f"  Lambda (penalty): {results['optimal_lambda']:.4f}")
    print(f"  C (inverse): {results['optimal_C']:.4f}")
    
    print(f"\nModel Performance (Test Set):")
    print(f"  Accuracy: {results['test_accuracy']:.3f}")
    print(f"  Precision: {results['test_precision']:.3f}")
    print(f"  Recall: {results['test_recall']:.3f}")
    print(f"  F1 Score: {results['test_f1']:.3f}")
    print(f"  AUC: {results['test_auc']:.3f}")
    
    print(f"\nOverfitting Check:")
    print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"  Gap: {results['train_accuracy'] - results['test_accuracy']:.3f}")
    
    print(f"\nTreatment Effects (Marginal Effects in pp):")
    for i, name in enumerate(results['feature_names'][:4]):
        effect = results['marginal_effects'][i] * 100
        print(f"  {name.replace('treatment_', '').title()}: {effect:+.2f} pp")
    
    print("="*60 + "\n")


def create_coefficient_and_error_plots(results, paths):
    """
    Create ISLP-style coefficient paths and CV error plots
    """
    # Extract coefficient paths from the model
    # coefs_paths_[1] has shape (n_folds, n_Cs, n_features)
    coef_paths_all_folds = results['model'].coefs_paths_[1]
    
    # Average across folds to get mean coefficient path
    coefficients_path = coef_paths_all_folds.mean(axis=0)  # Shape: (n_Cs, n_features)
    
    # Get CV scores
    cv_scores_all = results['model'].scores_[1]  # Shape: (n_folds, n_Cs)
    cv_means = cv_scores_all.mean(axis=0)
    cv_stds = cv_scores_all.std(axis=0)
    
    # Use the actual Cs that were used by the model
    actual_Cs = results['model'].Cs_
    actual_lambdas = 1 / actual_Cs
    
    print(f"Coefficient path shape: {coefficients_path.shape}")
    print(f"Number of lambdas: {len(actual_lambdas)}")
    print(f"Lambda range: {actual_lambdas.min():.4e} to {actual_lambdas.max():.4e}")
    
    # ========== PLOT 1: Coefficient Paths ==========
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    
    # Use distinct colors for each feature
    colors = plt.cm.tab20(np.linspace(0, 1, len(results['feature_names'])))
    
    # Plot each feature's coefficient path
    for i, feature in enumerate(results['feature_names']):
        if 'treatment' in feature:
            # Highlight treatment variables
            label = feature.replace('treatment_', '').title()
            linewidth = 2.5
            alpha = 1.0
        else:
            # Control variables with thinner lines
            label = feature
            linewidth = 1.5
            alpha = 0.6
        
        ax1.plot(-np.log(actual_lambdas), coefficients_path[:, i], 
                color=colors[i], label=label, linewidth=linewidth, alpha=alpha)
    
    # Add vertical line for optimal lambda
    ax1.axvline(-np.log(results['optimal_lambda']), c='red', ls='--', 
                linewidth=2, alpha=0.8, label=f'Optimal λ = {results["optimal_lambda"]:.4f}')
    
    # Formatting
    ax1.set_xlabel('$-\\log(\\lambda)$', fontsize=14)
    ax1.set_ylabel('Standardized Coefficients', fontsize=14)
    ax1.set_title('Ridge Regression: Coefficient Paths', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis limits to show full range
    max_abs_coef = np.abs(coefficients_path).max()
    ax1.set_ylim([-max_abs_coef*1.1, max_abs_coef*1.1])
    
    plt.tight_layout()
    plt.savefig(paths['plots'] + 'ridge_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 2: Cross-Validation Error ==========
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    # Convert negative log loss to positive for display
    cv_means_positive = -cv_means
    
    # Plot with error bars
    ax2.errorbar(-np.log(actual_lambdas), cv_means_positive,
                 yerr=cv_stds / np.sqrt(5),  # Standard error with 5 folds
                 fmt='o-', markersize=4, linewidth=2, 
                 capsize=3, capthick=1.5,
                 color='steelblue', ecolor='lightblue', alpha=0.8,
                 label='CV Error ± SE')
    
    # Fill between for confidence interval
    ax2.fill_between(-np.log(actual_lambdas),
                      cv_means_positive - cv_stds / np.sqrt(5),
                      cv_means_positive + cv_stds / np.sqrt(5),
                      alpha=0.2, color='steelblue')
    
    # Add vertical line for optimal lambda
    ax2.axvline(-np.log(results['optimal_lambda']), c='red', ls='--', 
                linewidth=2, alpha=0.8, label=f'Optimal λ = {results["optimal_lambda"]:.4f}')
    
    # Add horizontal line at minimum error
    min_idx = np.argmin(cv_means_positive)
    ax2.axhline(cv_means_positive[min_idx], c='green', ls=':', 
                linewidth=1.5, alpha=0.5, label='Minimum CV Error')
    
    # Formatting
    ax2.set_xlabel('$-\\log(\\lambda)$', fontsize=14)
    ax2.set_ylabel('Cross-Validation Log Loss', fontsize=14)
    ax2.set_title('Ridge Regression: Cross-Validation Error', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(paths['plots'] + 'ridge_cv_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved:")
    print(f"  - {paths['plots']}ridge_coefficients.png")
    print(f"  - {paths['plots']}ridge_cv_error.png")

# Run the analysis
run_ridge_analysis()
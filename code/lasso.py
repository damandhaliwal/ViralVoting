import numpy as np
import matplotlib.pyplot as plt
from data_clean import clean_data
from utils import get_project_paths
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


def run_lasso_analysis(plot_top_n=None):
    data = clean_data()
    paths = get_project_paths()

    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne', 'treatment_neighbors', 'treatment_self']
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'p2004', 'p2000', 'p2002']
    intensity_vars = ['treatment_intensity', 'high_block_intensity']
    feature_names = treatment_vars + control_vars + intensity_vars

    X = np.array(data[feature_names])
    y = np.array(data['voted'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    lambdas = np.logspace(-6, 6, 100)
    Cs = 1 / lambdas

    model = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        penalty='l1',
        solver='liblinear',
        scoring='neg_log_loss',
        random_state=42,
        max_iter=2000,
        refit=True
    )

    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_logloss = log_loss(y_test, y_test_proba)

    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    coefficients = model.coef_[0]
    optimal_C = model.C_[0]
    optimal_lambda = 1 / optimal_C

    cv_scores = model.scores_[1].mean(axis=0)
    best_cv_score = cv_scores[list(model.Cs_).index(optimal_C)]

    test_proba_mean = y_test_proba.mean()
    marginal_effects = coefficients * test_proba_mean * (1 - test_proba_mean)

    n_selected = np.sum(coefficients != 0)
    selected_features = [feature_names[i] for i in range(len(coefficients)) if coefficients[i] != 0]

    results = {
        'model': model,
        'optimal_C': optimal_C,
        'optimal_lambda': optimal_lambda,
        'coefficients': coefficients,
        'marginal_effects': marginal_effects,
        'feature_names': feature_names,
        'n_selected_features': n_selected,
        'selected_features': selected_features,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_logloss': test_logloss,
        'train_accuracy': train_accuracy,
        'train_auc': train_auc,
        'cv_best_score': best_cv_score,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_vote_rate': y_train.mean(),
        'test_vote_rate': y_test.mean(),
        'train_mean': train_mean,
        'train_std': train_std
    }

    _create_coefficient_plot(results, paths, plot_top_n)
    _print_results(results, 'LASSO')

    return results


def _print_results(results, method_name):
    print("\n" + "=" * 60)
    print(f"{method_name} REGRESSION RESULTS")
    print("=" * 60)
    print(f"Optimal λ: {results['optimal_lambda']:.6f}")
    print(f"Optimal C: {results['optimal_C']:.6f}")

    if 'n_selected_features' in results:
        print(f"Selected features: {results['n_selected_features']}/{len(results['feature_names'])}")

    print("\nTest Set Performance:")
    print(f"  Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Precision: {results['test_precision']:.4f}")
    print(f"  Recall:    {results['test_recall']:.4f}")
    print(f"  F1 Score:  {results['test_f1']:.4f}")
    print(f"  ROC AUC:   {results['test_auc']:.4f}")
    print(f"  Log Loss:  {results['test_logloss']:.4f}")

    print("\nTrain Set Performance:")
    print(f"  Accuracy:  {results['train_accuracy']:.4f}")
    print(f"  ROC AUC:   {results['train_auc']:.4f}")

    print("\nTreatment Effects (Coefficients):")
    for i, name in enumerate(results['feature_names']):
        if 'treatment' in name:
            coef = results['coefficients'][i]
            me = results['marginal_effects'][i]
            print(f"  {name:30s}: {coef:8.4f} (ME: {me:7.4f})")

    print("=" * 60)


def _create_coefficient_plot(results, paths, plot_top_n=None):
    coef_paths_all_folds = results['model'].coefs_paths_[1]
    coefficients_path = coef_paths_all_folds.mean(axis=0)

    actual_Cs = results['model'].Cs_
    actual_lambdas = 1 / actual_Cs

    if plot_top_n is not None:
        final_coefs = coefficients_path[-1, :]
        top_indices = np.argsort(np.abs(final_coefs))[-plot_top_n:]
        feature_indices = top_indices
        feature_labels = [results['feature_names'][i] for i in top_indices]
    else:
        feature_indices = range(len(results['feature_names']))
        feature_labels = results['feature_names']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    colors = plt.cm.tab20(np.linspace(0, 1, len(feature_indices)))

    for idx, (i, feature) in enumerate(zip(feature_indices, feature_labels)):
        if 'treatment' in feature:
            label = feature.replace('treatment_', '').title()
            linewidth = 2.5
            alpha = 1.0
        else:
            label = feature
            linewidth = 1.5
            alpha = 0.6

        ax.plot(-np.log(actual_lambdas), coefficients_path[:, i],
                color=colors[idx], label=label, linewidth=linewidth, alpha=alpha)

    ax.axvline(-np.log(results['optimal_lambda']), c='red', ls='--',
               linewidth=2, alpha=0.8, label=f'Optimal λ = {results["optimal_lambda"]:.4f}')

    ax.set_xlabel('$-\\log(\\lambda)$', fontsize=14)
    ax.set_ylabel('Standardized Coefficients', fontsize=14)
    ax.set_title('LASSO Regression: Coefficient Paths', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    max_abs_coef = np.abs(coefficients_path[:, feature_indices]).max()
    ax.set_ylim([-max_abs_coef * 1.2, max_abs_coef * 1.2])
    ax.set_xlim([np.log(actual_lambdas).min(), np.log(actual_lambdas).max()])

    plt.tight_layout()
    plt.savefig(paths['plots'] + 'figure4a.png', dpi=600, bbox_inches='tight')
    plt.close()

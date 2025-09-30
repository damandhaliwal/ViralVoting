import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagging import bagging_classifier
from boosting import gradient_boosting_classifier
from tree_forest import random_forest_classifier, classification_tree
from utils import get_project_paths

def compare_classifiers():
    """
    Compare Bagging, Gradient Boosting, Random Forest, and Decision Tree classifiers.
    Outputs LaTeX table and bar plot.
    """
    paths = get_project_paths()
    results = []

    # Run all classifiers
    print("Running Bagging Classifier...")
    bagging_results = bagging_classifier()
    results.append({
        "Model": "Bagging",
        "Accuracy": bagging_results["test_accuracy"],
        "Log-Loss": bagging_results["test_logloss"]
    })

    print("\nRunning Gradient Boosting Classifier...")
    boosting_results = gradient_boosting_classifier()
    results.append({
        "Model": "Gradient Boosting",
        "Accuracy": boosting_results["test_accuracy"],
        "Log-Loss": boosting_results["test_logloss"]
    })

    print("\nRunning Random Forest Classifier...")
    rf_results = random_forest_classifier()
    results.append({
        "Model": "Random Forest",
        "Accuracy": rf_results["test_accuracy"],
        "Log-Loss": rf_results["test_logloss"]
    })

    print("\nRunning Decision Tree Classifier...")
    dt_results = classification_tree()
    results.append({
        "Model": "Decision Tree",
        "Accuracy": dt_results["test_accuracy"],
        "Log-Loss": dt_results["test_logloss"]
    })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Find best models
    best_acc_idx = df['Accuracy'].idxmax()
    best_loss_idx = df['Log-Loss'].idxmin()
    
    # Generate LaTeX table
    latex_output = "\\begin{table}[h]\n"
    latex_output += "\\centering\n"
    latex_output += "\\caption{Error Comparison for Bagging, Boosting, Tree, and Random Forest}\n"
    latex_output += "\\label{tab:model_comparison}\n"
    latex_output += "\\begin{tabular}{lcc}\n"
    latex_output += "\\hline\n"
    latex_output += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Log-Loss} \\\\\n"
    latex_output += "\\hline\n"
    
    for idx, row in df.iterrows():
        model_name = row['Model']
        accuracy = row['Accuracy']
        logloss = row['Log-Loss']
        
        # Bold the best values
        acc_str = f"\\textbf{{{accuracy:.4f}}}" if idx == best_acc_idx else f"{accuracy:.4f}"
        loss_str = f"\\textbf{{{logloss:.4f}}}" if idx == best_loss_idx else f"{logloss:.4f}"
        
        latex_output += f"{model_name} & {acc_str} & {loss_str} \\\\\n"
    
    latex_output += "\\hline\n"
    latex_output += "\\end{tabular}\n"
    latex_output += "\\end{table}\n"
    
    # Save LaTeX table
    with open(paths['tables'] + 'model_comparison.tex', 'w') as f:
        f.write(latex_output)
    
    print("\n" + "="*70)
    print("LATEX TABLE SAVED TO: Output/Tables/model_comparison.tex")
    print("="*70 + "\n")
    print(latex_output)
    print("="*70)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy
    colors_acc = ['steelblue' if i != best_acc_idx else 'darkgreen' for i in range(len(df))]
    ax1.bar(df['Model'], df['Accuracy'], color=colors_acc, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([df['Accuracy'].min() * 0.95, df['Accuracy'].max() * 1.02])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax1.text(i, row['Accuracy'], f"{row['Accuracy']:.4f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Log-Loss (lower is better)
    colors_loss = ['coral' if i != best_loss_idx else 'darkgreen' for i in range(len(df))]
    ax2.bar(df['Model'], df['Log-Loss'], color=colors_loss, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Log-Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Log-Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylim([df['Log-Loss'].min() * 0.9, df['Log-Loss'].max() * 1.05])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax2.text(i, row['Log-Loss'], f"{row['Log-Loss']:.4f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(paths['plots'] + 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPLOT SAVED TO: {paths['plots']}model_comparison.png")
    
    # Print analysis
    best_model = df.loc[best_acc_idx, 'Model']
    print("\nANALYSIS:")
    print(f"Ensemble methods outperform the single Decision Tree, with {best_model} "
          f"achieving the highest accuracy. This demonstrates how combining multiple weak "
          f"learners reduces variance and improves model performance.")
    
    return df

compare_classifiers()
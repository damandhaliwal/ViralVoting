# Replication Package: Viral Voting

**A study in the information diffusion effects of different social pressure treatments on voter turnout**

## Data Source
Data obtained from Stanford repository: Gerber, Green, and Larimer (2008) field experiment on social pressure and voter turnout.

**Source:** https://github.com/gsbDBI/ExperimentData

## Usage
Run the main analysis script to generate all results:
```bash
python main.py
```

This will produce:
- Regression tables (OLS, LASSO, Ridge, SAR)
- Machine learning model comparisons (Tree, Forest, Bagging, Boosting, GNN)
- Visualizations (coefficient paths, feature importance, DAGs, spillover effects)
- All outputs saved to `tables/` and `plots/` directories

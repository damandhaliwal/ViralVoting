# s, x and t learners
# Daman Dhaliwal

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from data_clean import clean_data
from utils import get_project_paths

FEATURES = [
    "sex",
    "yob",
    "g2000",
    "g2002",
    "g2004",
    "p2000",
    "p2002",
    "p2004",
    "treatment_intensity",
    "high_block_intensity",
]

TREATMENT_COLUMNS = {
    "Civic Duty": "treatment_civic duty",
    "Hawthorne": "treatment_hawthorne",
    "Self": "treatment_self",
    "Neighbors": "treatment_neighbors",
}


def prepare_datasets() -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    data = clean_data()

    X_full = data[FEATURES].copy()
    X_full["yob"] = X_full["yob"] - X_full["yob"].mean()
    outcome = data["voted"].astype(float)
    control = data["treatment_control"].astype(int)

    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series]] = {}

    for pretty_name, column in TREATMENT_COLUMNS.items():
        mask = (data[column] == 1) | (control == 1)
        X = X_full.loc[mask].reset_index(drop=True)
        T = data.loc[mask, column].astype(int).reset_index(drop=True)
        y = outcome.loc[mask].reset_index(drop=True)
        datasets[pretty_name] = (X, T, y)

    return datasets


def _base_learner() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


@dataclass
class LearnerResult:
    name: str
    ate: float
    cate: np.ndarray


def run_s_learner(X: pd.DataFrame, T: pd.Series, y: pd.Series) -> LearnerResult:
    model = _base_learner()
    X_augmented = X.copy()
    X_augmented["treatment"] = T
    model.fit(X_augmented, y)
    y1 = model.predict(X_augmented.assign(treatment=1))
    y0 = model.predict(X_augmented.assign(treatment=0))
    cate = y1 - y0
    return LearnerResult(name="S-Learner", ate=cate.mean(), cate=cate)


def run_t_learner(X: pd.DataFrame, T: pd.Series, y: pd.Series) -> LearnerResult:
    model_treated = _base_learner()
    model_control = _base_learner()
    model_treated.fit(X[T == 1], y[T == 1])
    model_control.fit(X[T == 0], y[T == 0])
    mu1 = model_treated.predict(X)
    mu0 = model_control.predict(X)
    cate = mu1 - mu0
    return LearnerResult(name="T-Learner", ate=cate.mean(), cate=cate)


def run_x_learner(X: pd.DataFrame, T: pd.Series, y: pd.Series) -> LearnerResult:
    model_treated = _base_learner()
    model_control = _base_learner()
    model_treated.fit(X[T == 1], y[T == 1])
    model_control.fit(X[T == 0], y[T == 0])
    mu1 = model_treated.predict(X)
    mu0 = model_control.predict(X)

    D1 = y[T == 1] - mu0[T == 1]
    D0 = mu1[T == 0] - y[T == 0]

    tau1_model = _base_learner()
    tau0_model = _base_learner()
    tau1_model.fit(X[T == 1], D1)
    tau0_model.fit(X[T == 0], D0)

    tau1 = tau1_model.predict(X)
    tau0 = tau0_model.predict(X)

    propensity_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    propensity_model.fit(X, T)
    propensity = propensity_model.predict_proba(X)[:, 1]

    cate = propensity * tau0 + (1 - propensity) * tau1
    return LearnerResult(name="X-Learner", ate=cate.mean(), cate=cate)


def generate_table8(datasets):
    results_list = []

    for treatment_name, (X, T, y) in datasets.items():
        s_res = run_s_learner(X, T, y)
        t_res = run_t_learner(X, T, y)
        x_res = run_x_learner(X, T, y)

        # Calculate Pairwise CATE Differences (Mean Absolute Error)
        diff_s_t = np.abs(s_res.cate - t_res.cate).mean()
        diff_s_x = np.abs(s_res.cate - x_res.cate).mean()
        diff_t_x = np.abs(t_res.cate - x_res.cate).mean()

        results_list.append({
            'Treatment': treatment_name,
            'S-Learner': s_res.ate,
            'T-Learner': t_res.ate,
            'X-Learner': x_res.ate,
            'S vs T': diff_s_t,
            'S vs X': diff_s_x,
            'T vs X': diff_t_x
        })

    df_table8 = pd.DataFrame(results_list)

    paths = get_project_paths()

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_table8.to_latex(
        output_dir + 'table8.tex',
        index=False,
        float_format="%.4f",
        caption="Summary of Treatment Effect Estimates",
        label="tab:learner_summary"
    )


def generate_table9(datasets):
    # Analyzing heterogeneity using X-Learner (as per image description)
    labels = ["Low", "Mid", "High"]
    intensity_results = []

    for treatment_name, (X, T, y) in datasets.items():
        # Run X-Learner only
        result = run_x_learner(X, T, y)

        # Bin by intensity
        bins = pd.qcut(
            X["treatment_intensity"],
            q=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            labels=labels,
            duplicates="drop",
        )

        summary = (
            pd.DataFrame({"bin": bins, "cate": result.cate})
            .groupby("bin", observed=False)["cate"]
            .mean()
        )

        intensity_results.append({
            'Treatment Arm': treatment_name,
            'Low': summary.get('Low', np.nan),
            'Mid': summary.get('Mid', np.nan),
            'High': summary.get('High', np.nan)
        })

    df_table9 = pd.DataFrame(intensity_results)

    paths = get_project_paths()

    output_dir = paths['tables']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_table9.to_latex(
        output_dir + 'table9.tex',
        index=False,
        float_format="%.4f",
        caption="ATE By Treatment Intensity (X-Learner)",
        label="tab:learner_intensity"
    )


def main():
    datasets = prepare_datasets()

    generate_table8(datasets)
    generate_table9(datasets)


if __name__ == "__main__":
    main()
"""
Hyperparameter optimization and rigorous cross-validation for downstream classifier.

This script supports two modes:
  --mode hpo    : Run Optuna to find best C for both I-JEPA and ResNet-50 embeddings.
  --mode cv     : Run 10 repeats x 10 folds stratified CV using the best C from HPO.

AI-assist prompt:
"Write a script that loads precomputed embeddings, uses Optuna to search the
regularization parameter C of a LogisticRegression inside a sklearn Pipeline,
then runs a rigorous 10x10 repeated stratified k-fold cross-validation and
performs a Wilcoxon signed-rank test between two models. Save all results to CSV."
"""

import argparse
import json
import os
import pickle

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import yaml


def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]
    return X, y


def build_pipeline(C: float):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            random_state=42,
            n_jobs=5,
        )),
    ])


def objective(trial, X, y, n_splits=5, n_repeats=1):
    C = trial.suggest_float("C", 1e-4, 1e2, log=True)
    clf = build_pipeline(C)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=5)
    return scores.mean()


def run_hpo(embeddings_a_path: str, embeddings_b_path: str, n_trials: int = 50):
    X_a, y_a = load_embeddings(embeddings_a_path)
    X_b, y_b = load_embeddings(embeddings_b_path)

    results = {}
    for name, X, y in [("ijepa", X_a, y_a), ("resnet50", X_b, y_b)]:
        study = optuna.create_study(direction="maximize", study_name=f"hpo_{name}")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, show_progress_bar=True)
        results[name] = {
            "best_C": study.best_params["C"],
            "best_score": study.best_value,
            "study": study,
        }
        print(f"[{name}] Best C: {study.best_params['C']:.6f} | F1-macro: {study.best_value:.4f}")
    return results


def run_cv(embeddings_a_path: str, embeddings_b_path: str, best_C_a: float, best_C_b: float, n_repeats=10, n_folds=10):
    X_a, y_a = load_embeddings(embeddings_a_path)
    X_b, y_b = load_embeddings(embeddings_b_path)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    clf_a = build_pipeline(best_C_a)
    scores_a = cross_val_score(clf_a, X_a, y_a, cv=cv, scoring="f1_macro", n_jobs=5)

    clf_b = build_pipeline(best_C_b)
    scores_b = cross_val_score(clf_b, X_b, y_b, cv=cv, scoring="f1_macro", n_jobs=5)

    # Wilcoxon signed-rank test (paired by fold index)
    stat, p_value = wilcoxon(scores_a, scores_b)

    df = pd.DataFrame({
        "fold_idx": np.arange(len(scores_a)),
        "ijepa_f1_macro": scores_a,
        "resnet50_f1_macro": scores_b,
    })

    print(f"[CV] I-JEPA mean F1: {scores_a.mean():.4f} +/- {scores_a.std():.4f}")
    print(f"[CV] ResNet-50 mean F1: {scores_b.mean():.4f} +/- {scores_b.std():.4f}")
    print(f"[Wilcoxon] stat={stat}, p-value={p_value:.6f}")

    return df, stat, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["hpo", "cv"])
    parser.add_argument("--embeddings_a", default="data/processed/ijepa_embeddings.npz")
    parser.add_argument("--embeddings_b", default="data/processed/resnet50_embeddings.npz")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--best_C_a", type=float, default=None)
    parser.add_argument("--best_C_b", type=float, default=None)
    parser.add_argument("--output_dir", default="reports/hpo")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "hpo":
        results = run_hpo(args.embeddings_a, args.embeddings_b, n_trials=args.n_trials)
        # Save studies
        for name, res in results.items():
            with open(f"{args.output_dir}/study_{name}.pkl", "wb") as f:
                pickle.dump(res["study"], f)
        with open(f"{args.output_dir}/best_params.json", "w") as f:
            json.dump({k: {"best_C": v["best_C"], "best_score": v["best_score"]} for k, v in results.items()}, f, indent=2)
    elif args.mode == "cv":
        if args.best_C_a is None or args.best_C_b is None:
            # Try to load from HPO output
            with open(f"{args.output_dir}/best_params.json", "r") as f:
                best = json.load(f)
            args.best_C_a = best["ijepa"]["best_C"]
            args.best_C_b = best["resnet50"]["best_C"]
        df, stat, p_value = run_cv(
            args.embeddings_a, args.embeddings_b,
            args.best_C_a, args.best_C_b,
            n_repeats=args.n_repeats, n_folds=args.n_folds,
        )
        df.to_csv(f"{args.output_dir}/cv_scores.csv", index=False)
        with open(f"{args.output_dir}/wilcoxon_results.json", "w") as f:
            json.dump({"statistic": float(stat), "p_value": float(p_value)}, f, indent=2)


if __name__ == "__main__":
    main()

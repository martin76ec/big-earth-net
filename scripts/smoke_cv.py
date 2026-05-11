"""
Smoke test: verify RepeatedStratifiedKFold + Wilcoxon code runs without errors.

AI-assist prompt:
"Write a smoke test that runs 2 repeats x 2 folds CV on synthetic data with
LogisticRegression and then runs a Wilcoxon signed-rank test."
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon


def main():
    X = np.random.randn(200, 128)
    y = np.random.randint(0, 5, size=200)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs")),
    ])

    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    print(f"Scores ({len(scores)}): {scores}")

    # Dummy paired scores for Wilcoxon
    scores_b = scores + np.random.randn(len(scores)) * 0.05
    stat, p = wilcoxon(scores, scores_b)
    print(f"Wilcoxon: stat={stat}, p={p:.4f}")
    print("[smoke_cv] PASSED")


if __name__ == "__main__":
    main()

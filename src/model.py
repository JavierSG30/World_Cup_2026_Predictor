"""
World Cup 2026 Predictor — Model Training & Evaluation
--------------------------------------------------------
Switch MODEL_CHOICE at the top to select which model to train.
All models are saved to models/ and are drop-in compatible
with predict.py and matchup.py.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (log_loss, accuracy_score,
                                     confusion_matrix, classification_report)

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── CHOOSE YOUR MODEL HERE ───────────────────────────────────────────────────
#
#   "logistic"  — Multinomial logistic regression (L1/L2/ElasticNet compared)
#   "forest"    — Random Forest (no scaling needed)
#   "gradient"  — Gradient Boosting (no scaling needed)
#
MODEL_CHOICE = "forest"

# ── CLASS WEIGHTING ──────────────────────────────────────────────────────────
# Penalises draw misclassifications more heavily during training.
#
#   None        — no weighting (current behaviour)
#   "moderate"  - Lighter weighting than "balanced" to force realistic draw prob
#   "balanced"  — auto-weights inversely proportional to class frequency
#                 Draw ≈1.52×, Away Win ≈1.11×, Home Win ≈0.69×
#   "aggressive"— manually amplified draw weight, stronger push toward draws
#                 Draw ≈3×, Away Win ≈1×, Home Win ≈1×
#
# Note: class_weight is NOT supported by GradientBoostingClassifier.
# It is ignored if MODEL_CHOICE = "gradient".
#
CLASS_WEIGHT = None
#
# ══════════════════════════════════════════════════════════════════════════════

ID_COLS = ["date", "home_team", "away_team", "tournament"]
TARGET  = "result"

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_splits():
    train = pd.read_csv(PROCESSED / "train.csv", parse_dates=["date"])
    val   = pd.read_csv(PROCESSED / "val.csv",   parse_dates=["date"])
    test  = pd.read_csv(PROCESSED / "test.csv",  parse_dates=["date"])

    feat_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]

    X_train, y_train = train[feat_cols], train[TARGET]
    X_val,   y_val   = val[feat_cols],   val[TARGET]
    X_test,  y_test  = test[feat_cols],  test[TARGET]

    print(f"Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    print(f"Features ({len(feat_cols)}): {feat_cols}\n")
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(name, model, X, y, split_name="val"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    acc    = accuracy_score(y, y_pred)
    ll     = log_loss(y, y_prob)
    cm     = confusion_matrix(y, y_pred, labels=[2, 1, 0])

    print(f"\n{'─'*55}")
    print(f"  {name}  [{split_name}]")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Log loss : {ll:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Home Win   Draw   Away Win")
    for i, label in enumerate(["Home Win", "Draw", "Away Win"]):
        print(f"  {label:<12}  {cm[i,0]:>6}  {cm[i,1]:>6}  {cm[i,2]:>8}")
    print(f"\n  Classification report:")
    print(classification_report(y, y_pred,
                                target_names=["Away Win","Draw","Home Win"],
                                digits=3, zero_division=0))
    return {"accuracy": acc, "log_loss": ll}

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_class_weight():
    if CLASS_WEIGHT is None:
        return None
    elif CLASS_WEIGHT == "balanced":
        return "balanced"
    elif CLASS_WEIGHT == "moderate":
        return {0: 1.0, 1: 1.1, 2: 1.0}   # draw gets a gentle nudge
    elif CLASS_WEIGHT == "aggressive":
        return {0: 1.0, 1: 3.0, 2: 1.0}
    else:
        raise ValueError(f"Unknown CLASS_WEIGHT: {CLASS_WEIGHT}")


def train_logistic(X_train, y_train, X_val, y_val):
    """
    Trains L1, L2, and ElasticNet variants and picks the best on val log loss.
    Requires StandardScaler (logistic regression is sensitive to feature scale).
    """
    cw = resolve_class_weight()
    print(f"  Class weight: {CLASS_WEIGHT} → {cw}")

    variants = [
        ("L2",         {"penalty": "l2",         "l1_ratio": 0.0}),
        ("L1",         {"penalty": "l1",          "l1_ratio": 1.0}),
        ("ElasticNet", {"penalty": "elasticnet",  "l1_ratio": 0.5}),
    ]

    results = {}
    for vname, kwargs in variants:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                           solver="saga", C=1.0,
                           max_iter=2000, random_state=42,
                           class_weight=cw,
                           **kwargs))
        ])
        model.fit(X_train, y_train)
        ll  = log_loss(y_val, model.predict_proba(X_val))
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  Logistic ({vname:<12})  val log loss: {ll:.4f}  acc: {acc:.4f}")
        results[vname] = {"model": model, "log_loss": ll}

    best_name = min(results, key=lambda k: results[k]["log_loss"])
    print(f"\n  → Best logistic variant: {best_name}")
    return results[best_name]["model"], f"Logistic ({best_name})"

def train_forest(X_train, y_train):
    """
    Random Forest — ensemble of decision trees, each trained on a random
    subset of rows and features. No scaling needed (trees are scale-invariant).

    Key hyperparameters:
      n_estimators     : number of trees (more = more stable, slower)
      max_depth        : maximum depth per tree (controls overfitting)
      min_samples_leaf : minimum matches in a leaf (smooths probabilities)
      max_features     : features considered per split ('sqrt' = sqrt(25) ≈ 5)
    """
    cw = resolve_class_weight()
    print(f"  Class weight: {CLASS_WEIGHT} → {cw}")

    model = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 8,
        min_samples_leaf = 10,
        max_features     = "sqrt",
        random_state     = 42,
        n_jobs           = -1,
        class_weight     = cw,
    )
    model.fit(X_train, y_train)
    print(f"  Random Forest trained — {model.n_estimators} trees")
    return model, "Random Forest"


def train_gradient(X_train, y_train, X_val, y_val):
    """
    Gradient Boosting — sequential ensemble where each tree corrects the
    errors of all previous trees. Slower than Random Forest but often more
    accurate. No scaling needed.

    Key hyperparameters:
      n_estimators  : number of boosting rounds
      learning_rate : shrinkage applied to each tree's contribution
      max_depth     : depth per tree (shallower = more regularised)
      subsample     : fraction of training rows used per tree
    """
    model = GradientBoostingClassifier(
        n_estimators  = 300,
        learning_rate = 0.05,
        max_depth     = 4,
        subsample     = 0.8,
        random_state  = 42,
    )
    model.fit(X_train, y_train)
    print(f"  Gradient Boosting trained — {model.n_estimators} trees")
    return model, "Gradient Boosting"

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_importance(model, feat_cols, model_name):
    """Works for Random Forest and Gradient Boosting (both have feature_importances_).
       For logistic regression, prints standardised coefficients instead."""

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        scores = model.feature_importances_
        ranked = sorted(zip(feat_cols, scores), key=lambda x: x[1], reverse=True)
        print(f"\n  Feature importance (mean decrease in impurity) — top 12:")
        print(f"  {'Feature':<35} {'Score':>7}  Chart")
        print(f"  {'─'*35} {'─'*7}  {'─'*30}")
        for fname, score in ranked[:12]:
            bar = "█" * int(score * 400)
            print(f"  {fname:<35} {score:.4f}  {bar}")

    elif hasattr(model, "named_steps"):
        # Logistic regression pipeline
        clf   = model.named_steps["clf"]
        coefs = clf.coef_
        max_abs = np.max(np.abs(coefs), axis=0)
        order   = np.argsort(max_abs)[::-1]

        print(f"\n  Standardised coefficients — top 12:")
        print(f"  {'Feature':<35} {'Away Win':>9} {'Draw':>9} {'Home Win':>9}  {'|max|':>6}")
        print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*9}  {'─'*6}")
        for i in order[:12]:
            vals = coefs[:, i]
            print(f"  {feat_cols[i]:<35} {vals[0]:>+9.3f} {vals[1]:>+9.3f} "
                  f"{vals[2]:>+9.3f}  {max_abs[i]:>6.3f}")

        zeroed = [feat_cols[i] for i in range(len(feat_cols))
                  if np.all(np.abs(coefs[:, i]) < 1e-6)]
        if zeroed:
            print(f"\n  Features zeroed out by regularisation:")
            for f in zeroed:
                print(f"    - {f}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"=== Model choice: {MODEL_CHOICE} ===\n")

    print("=== Loading data ===")
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_splits()



    print(f"=== Training ===")
    if MODEL_CHOICE == "logistic":
        model, model_name = train_logistic(X_train, y_train, X_val, y_val)
    elif MODEL_CHOICE == "forest":
        model, model_name = train_forest(X_train, y_train)
    elif MODEL_CHOICE == "gradient":
        model, model_name = train_gradient(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown MODEL_CHOICE: '{MODEL_CHOICE}'. "
                         f"Choose 'logistic', 'forest', or 'gradient'.")

    print(f"\n=== Validation results ===")
    val_metrics = evaluate(model_name, model, X_val, y_val, "val")

    print(f"\n=== Feature importance ===")
    print_feature_importance(model, feat_cols, model_name)

    print(f"\n=== Test results (held-out — run once) ===")
    evaluate(model_name, model, X_test, y_test, "test")

    print(f"\n=== Saving model ===")
    with open(MODELS / "logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS / "feature_cols.pkl", "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"Saved to {MODELS}/  (as logistic_model.pkl — used by predict.py)")
    print(f"Model type saved: {model_name}")
    print(f"Val accuracy: {val_metrics['accuracy']:.4f}  |  "
          f"Val log loss: {val_metrics['log_loss']:.4f}")

if __name__ == "__main__":
    main()

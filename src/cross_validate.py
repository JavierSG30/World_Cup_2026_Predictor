"""
World Cup 2026 Predictor — Time Series Cross-Validation
---------------------------------------------------------
Evaluates model stability across three independent time windows.
Elo features are computed on the full history (no leakage) but
each fold trains/validates/tests on a non-overlapping time chunk.

Run:
    python src/cross_validate.py

Outputs:
  - Per-fold metrics (accuracy, log loss, draw recall)
  - Average and std across folds
  - Comparison to single 70/15/15 split baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (log_loss, accuracy_score,
                                     confusion_matrix, classification_report)

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ══════════════════════════════════════════════════════════════════════════════
# ── SETTINGS ─────────────────────────────────────────────────────────────────
#
#   "logistic"  — L1 logistic regression
#   "forest"    — Random Forest
#
CV_MODEL      = "forest"
CV_CLASS_WT   = "balanced"   # None / "balanced" / "aggressive"
#
# ══════════════════════════════════════════════════════════════════════════════

ID_COLS = ["date", "home_team", "away_team", "tournament"]
TARGET  = "result"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FULL FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def load_featured():
    df = pd.read_csv(PROCESSED / "matches_featured.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    print(f"Loaded {len(df):,} matches | {len(feat_cols)} features")
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE THREE FIXED-WINDOW FOLDS
# ══════════════════════════════════════════════════════════════════════════════

def make_folds(df):
    """
    Split the full feature matrix into 3 equal thirds by match count.
    Within each third, apply 70/15/15 time split.
    Returns list of (train, val, test) DataFrames.
    """
    n  = len(df)
    t1 = n // 3
    t2 = 2 * (n // 3)

    thirds = [
        df.iloc[0:t1].reset_index(drop=True),
        df.iloc[t1:t2].reset_index(drop=True),
        df.iloc[t2:n].reset_index(drop=True),
    ]

    folds = []
    for i, chunk in enumerate(thirds):
        sz = len(chunk)
        i1 = int(sz * 0.70)
        i2 = int(sz * 0.85)
        train = chunk.iloc[:i1]
        val   = chunk.iloc[i1:i2]
        test  = chunk.iloc[i2:]
        folds.append((train, val, test))

        print(f"  Fold {i+1} ({sz} matches):")
        print(f"    Train: {len(train):4d}  "
              f"({train.iloc[0]['date'].date()} → {train.iloc[-1]['date'].date()})")
        print(f"    Val:   {len(val):4d}  "
              f"({val.iloc[0]['date'].date()} → {val.iloc[-1]['date'].date()})")
        print(f"    Test:  {len(test):4d}  "
              f"({test.iloc[0]['date'].date()} → {test.iloc[-1]['date'].date()})")

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def resolve_class_weight(setting):
    if setting is None:       return None
    if setting == "balanced": return "balanced"
    if setting == "aggressive": return {0: 1.0, 1: 3.0, 2: 1.0}
    raise ValueError(f"Unknown class weight: {setting}")


def build_model():
    cw = resolve_class_weight(CV_CLASS_WT)

    if CV_MODEL == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                           solver="saga", C=1.0, l1_ratio=1.0,
                           max_iter=2000, random_state=42,
                           class_weight=cw))
        ])
    elif CV_MODEL == "forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=10, max_features="sqrt",
            random_state=42, n_jobs=-1,
            class_weight=cw,
        )
    else:
        raise ValueError(f"Unknown CV_MODEL: {CV_MODEL}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE ONE FOLD
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fold(fold_num, train, val, test, feat_cols):
    X_train = train[feat_cols];  y_train = train[TARGET]
    X_val   = val[feat_cols];    y_val   = val[TARGET]
    X_test  = test[feat_cols];   y_test  = test[TARGET]

    model = build_model()
    model.fit(X_train, y_train)

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        acc    = accuracy_score(y, y_pred)
        ll     = log_loss(y, y_prob)
        cm     = confusion_matrix(y, y_pred, labels=[2, 1, 0])

        # Draw recall: row index 1 in labels=[2,1,0] → index 1 = Draw row
        draw_row    = cm[1]   # [predicted_HW, predicted_D, predicted_AW]
        draw_recall = draw_row[1] / draw_row.sum() if draw_row.sum() > 0 else 0

        results[split_name] = {
            "accuracy":     acc,
            "log_loss":     ll,
            "draw_recall":  draw_recall,
            "n_matches":    len(y),
            "cm":           cm,
        }

        print(f"    {split_name:5s}: acc={acc:.4f}  log_loss={ll:.4f}  "
              f"draw_recall={draw_recall:.3f}  (n={len(y)})")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. AGGREGATE AND PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    print(f"\n{'═'*65}")
    print(f"  CROSS-VALIDATION SUMMARY  "
          f"({CV_MODEL}, class_weight={CV_CLASS_WT})")
    print(f"{'═'*65}")

    for split in ["val", "test"]:
        accs   = [r[split]["accuracy"]    for r in all_results]
        lls    = [r[split]["log_loss"]    for r in all_results]
        drs    = [r[split]["draw_recall"] for r in all_results]

        print(f"\n  {split.upper()} SET:")
        print(f"  {'─'*55}")
        print(f"  {'Metric':<18} {'Fold 1':>8} {'Fold 2':>8} "
              f"{'Fold 3':>8}  {'Mean':>8} {'Std':>7}")
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8}  {'─'*8} {'─'*7}")

        for label, vals in [("Accuracy", accs),
                             ("Log loss", lls),
                             ("Draw recall", drs)]:
            mean = np.mean(vals)
            std  = np.std(vals)
            vals_str = "  ".join(f"{v:>8.4f}" for v in vals)
            print(f"  {label:<18} {vals_str}  {mean:>8.4f} {std:>7.4f}")

    print(f"\n  INTERPRETATION:")
    test_accs = [r["test"]["accuracy"] for r in all_results]
    test_lls  = [r["test"]["log_loss"] for r in all_results]
    print(f"  Mean test accuracy : {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  Mean test log loss : {np.mean(test_lls):.4f} ± {np.std(test_lls):.4f}")

    std_acc = np.std(test_accs)
    if std_acc < 0.02:
        verdict = "Low variance across folds — model is stable across time periods."
    elif std_acc < 0.04:
        verdict = "Moderate variance — some sensitivity to time period."
    else:
        verdict = "High variance — model performance is unstable across time periods."
    print(f"\n  Verdict: {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Loading feature matrix ===")
    df, feat_cols = load_featured()

    print("\n=== Defining folds ===")
    folds = make_folds(df)

    print(f"\n=== Running cross-validation "
          f"({CV_MODEL}, class_weight={CV_CLASS_WT}) ===")
    all_results = []
    for i, (train, val, test) in enumerate(folds, 1):
        print(f"\n  Fold {i}:")
        results = evaluate_fold(i, train, val, test, feat_cols)
        all_results.append(results)

    print_summary(all_results)

if __name__ == "__main__":
    main()

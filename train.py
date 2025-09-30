"""
Train IsolationForest on features.csv
- Uses normal + ambiguous as training baseline
- Scores all rows
- Outputs data/predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def main(args):
    df = pd.read_csv(args.in_csv)

    # Feature columns
    feat_cols = [
        "length","token_count","keyword_flag","entropy",
        "num_special_chars","upper_ratio","repeat_word_ratio",
        "time_delta_session","hour"
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    df[feat_cols] = df[feat_cols].fillna(0)

    # ✅ Train only on normal + ambiguous
    train_df = df[df['label'].isin(["normal", "ambiguous"])]
    X_train = train_df[feat_cols].values

    iso = IsolationForest(
        contamination=0.10,  # expected anomaly fraction; tune this
        random_state=42
    )
    iso.fit(X_train)

    # Predict on all
    X_all = df[feat_cols].values
    df["anomaly_score"] = iso.decision_function(X_all)
    df["anomaly_label"] = iso.predict(X_all)  # 1=normal, -1=anomaly

    # ✅ Tighten threshold manually
    threshold = -0.02   # tune this!
    df["threshold_label"] = (df["anomaly_score"] < threshold).astype(int)  # 1=attack, 0=benign

    # ✅ Add rule layer (keyword + long text)
    def rule_based_flag(row):
        if row["keyword_flag"] == 1 and row["length"] > 80:
            return 1  # attack
        return row["threshold_label"]  # fallback to threshold decision
    df["final_label"] = df.apply(rule_based_flag, axis=1)

    # Save predictions
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions -> {args.out_csv}")

    # Ground truth
    y_true = df["label"].apply(lambda x: 1 if x=="attack" else 0).values

    # Eval 1: raw IsolationForest
    y_pred_iso = (df["anomaly_label"] == -1).astype(int)
    print("\n=== Raw IsolationForest ===")
    print(confusion_matrix(y_true, y_pred_iso))
    print(classification_report(y_true, y_pred_iso, target_names=["benign","attack"]))

    # Eval 2: thresholded + rules
    y_pred_final = df["final_label"].values
    print("\n=== Threshold + Rules (final) ===")
    print(confusion_matrix(y_true, y_pred_final))
    print(classification_report(y_true, y_pred_final, target_names=["benign","attack"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv", default="data/features.csv")
    parser.add_argument("--out", dest="out_csv", default="data/predictions.csv")
    args = parser.parse_args()
    main(args)

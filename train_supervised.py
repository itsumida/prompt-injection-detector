"""
Train RandomForest classifier on features.csv
- Uses all labeled data (benign vs attack)
- Evaluates precision/recall
- Outputs data/predictions_supervised.csv
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.in_csv)

    # Feature columns (same as before)
    feat_cols = [
        "length","token_count","keyword_flag","entropy",
        "num_special_chars","upper_ratio","repeat_word_ratio",
        "time_delta_session","hour",
        # include any new features you engineered:
        "digit_ratio","special_ratio","ignore_flag","reveal_flag","sql_flag","auth_flag","whitespace_ratio"
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Encode labels (attack=1, benign=0)
    y = df["label"].apply(lambda x: 1 if x=="attack" else 0).values
    X = df[feat_cols].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"  # help with imbalance
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluation
    print("\n=== RandomForest Supervised Model ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["benign","attack"]))

    # Save predictions with scores
    df["predicted"] = clf.predict(X)
    df["attack_prob"] = clf.predict_proba(X)[:,1]
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions -> {args.out_csv}")

    import joblib
    joblib.dump(clf, "rf_model.pkl")
    print("Saved model -> rf_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv", default="data/features.csv")
    parser.add_argument("--out", dest="out_csv", default="data/predictions_supervised.csv")
    args = parser.parse_args()
    main(args)

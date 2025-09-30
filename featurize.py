# featurize.py
"""
Read data/prompt_dataset.csv -> compute features -> write data/features.csv
Usage:
  python featurize.py --in data/prompt_dataset.csv --out data/features.csv
"""

import argparse, re, collections, math
from datetime import datetime
import pandas as pd
import numpy as np

# --- Helpers ---
RE_NON_ALNUM = re.compile(r'[^a-z0-9]', re.IGNORECASE)
RE_WORDS = re.compile(r'\w+')

SUSPICIOUS_KEYWORDS = [
    "ignore","system","reveal","secret","password","token","jailbreak","print","drop","key",
    "admin","credential","credentials","api key","apikey","private key"
]

def normalized(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def keyword_flag(s):
    n = normalized(s)
    return int(any(k.replace(" ", "") in n for k in SUSPICIOUS_KEYWORDS))

def entropy(s):
    if not s:
        return 0.0
    counts = collections.Counter(s)
    probs = [v/len(s) for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def num_special_chars(s):
    return len(RE_NON_ALNUM.findall(s))

def upper_ratio(s):
    if len(s) == 0: return 0.0
    uppers = sum(1 for c in s if c.isupper())
    return uppers / len(s)

def repeat_word_ratio(s):
    tokens = RE_WORDS.findall(s.lower())
    if not tokens:
        return 0.0
    counts = collections.Counter(tokens)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(tokens)

def parse_ts(ts):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return pd.to_datetime(ts)
        except Exception:
            return None

# --- Main ---
def featurize(in_csv, out_csv):
    df = pd.read_csv(in_csv)

    # Ensure columns exist
    if 'prompt_text' not in df.columns:
        raise SystemExit("Input CSV must contain 'prompt_text' column")
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.NaT

    # === Basic features ===
    df['length'] = df['prompt_text'].astype(str).str.len()
    df['token_count'] = df['prompt_text'].astype(str).str.split().apply(len)
    df['keyword_flag'] = df['prompt_text'].fillna("").astype(str).apply(keyword_flag)
    df['entropy'] = df['prompt_text'].fillna("").astype(str).apply(entropy)
    df['num_special_chars'] = df['prompt_text'].fillna("").astype(str).apply(num_special_chars)
    df['upper_ratio'] = df['prompt_text'].fillna("").astype(str).apply(upper_ratio)
    df['repeat_word_ratio'] = df['prompt_text'].fillna("").astype(str).apply(repeat_word_ratio)

    # === Keyword-specific flags ===
    df["ignore_flag"] = df["prompt_text"].str.contains(r"\bignore\b", case=False).astype(int)
    df["reveal_flag"] = df["prompt_text"].str.contains(r"\breveal|secret|show\b", case=False).astype(int)
    df["sql_flag"] = df["prompt_text"].str.contains(r"\bdrop|select|insert|delete\b", case=False).astype(int)
    df["auth_flag"] = df["prompt_text"].str.contains(r"\bpassword|token|apikey|credential\b", case=False).astype(int)

    # === Ratios ===
    df["digit_ratio"] = df["prompt_text"].str.count(r"\d") / df["length"].replace(0,1)
    df["special_ratio"] = df["num_special_chars"] / df["length"].replace(0,1)
    df["whitespace_ratio"] = df["prompt_text"].str.count(" ") / df["token_count"].replace(0,1)

    # === Session features ===
    df['parsed_ts'] = df['timestamp'].apply(lambda x: parse_ts(x) if pd.notna(x) else None)
    df['parsed_ts'] = df['parsed_ts'].apply(lambda x: x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x)

    if 'user_id' in df.columns:
        df = df.sort_values(['user_id','parsed_ts'])
        prev_ts = {}
        deltas = []
        for _, row in df.iterrows():
            user = row['user_id']
            ts = row['parsed_ts']
            if user not in prev_ts or ts is pd.NaT or ts is None:
                deltas.append(np.nan)
            else:
                try:
                    delta = (ts - prev_ts[user]).total_seconds()
                    deltas.append(delta if delta >= 0 else np.nan)
                except Exception:
                    deltas.append(np.nan)
            prev_ts[user] = ts
        df['time_delta_session'] = deltas
    else:
        df['time_delta_session'] = np.nan

    # Hour of day
    df['hour'] = df['parsed_ts'].apply(lambda x: x.hour if (x is not None and not pd.isna(x)) else np.nan)

    # === Output columns ===
    out_cols = [
        'prompt_text','user_id','timestamp','label',
        'length','token_count','keyword_flag','entropy',
        'num_special_chars','upper_ratio','repeat_word_ratio',
        'time_delta_session','hour',
        'digit_ratio','special_ratio','ignore_flag','reveal_flag',
        'sql_flag','auth_flag','whitespace_ratio'
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    df[out_cols].to_csv(out_csv, index=False)
    print(f"Wrote features to {out_csv}")
    print("Sample rows:")
    print(df[out_cols].head(10).to_string(index=False))
    if 'label' in df.columns:
        print("\nLabel counts:")
        print(df['label'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize prompt dataset")
    parser.add_argument("--in", dest="in_csv", default="data/prompt_dataset.csv")
    parser.add_argument("--out", dest="out_csv", default="data/features.csv")
    args = parser.parse_args()
    featurize(args.in_csv, args.out_csv)

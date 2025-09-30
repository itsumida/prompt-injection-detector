import streamlit as st
import pandas as pd
import re, collections, math
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Feature helpers ---
RE_NON_ALNUM = re.compile(r'[^a-z0-9]', re.IGNORECASE)
RE_WORDS = re.compile(r'\w+')

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

def extract_features(text: str):
    length = len(text)
    token_count = len(text.split())
    keyword_flag = int(any(k in text.lower() for k in ["ignore","system","reveal","secret","password","token","jailbreak","print","drop","key","admin","credential","apikey","private key"]))
    entropy_val = entropy(text)
    num_special = num_special_chars(text)
    upper_val = upper_ratio(text)
    repeat_val = repeat_word_ratio(text)
    digit_ratio = sum(c.isdigit() for c in text) / max(1, length)
    special_ratio = num_special / max(1, length)
    whitespace_ratio = text.count(" ") / max(1, token_count)

    ignore_flag = int(re.search(r"\bignore\b", text, re.IGNORECASE) is not None)
    reveal_flag = int(re.search(r"\b(reveal|secret|show)\b", text, re.IGNORECASE) is not None)
    sql_flag = int(re.search(r"\b(drop|select|insert|delete)\b", text, re.IGNORECASE) is not None)
    auth_flag = int(re.search(r"\b(password|token|apikey|credential)\b", text, re.IGNORECASE) is not None)

    return pd.DataFrame([{
        "length": length,
        "token_count": token_count,
        "keyword_flag": keyword_flag,
        "entropy": entropy_val,
        "num_special_chars": num_special,
        "upper_ratio": upper_val,
        "repeat_word_ratio": repeat_val,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "whitespace_ratio": whitespace_ratio,
        "ignore_flag": ignore_flag,
        "reveal_flag": reveal_flag,
        "sql_flag": sql_flag,
        "auth_flag": auth_flag,
        "time_delta_session": 0,  # placeholder
        "hour": 12                # placeholder
    }])

# --- Load model (train_supervised.py must save one) ---
@st.cache_resource
def load_model():
    # if you want persistence: after training run:
    # joblib.dump(clf, "rf_model.pkl")
    try:
        clf = joblib.load("rf_model.pkl")
    except:
        st.warning("No saved model found. Retraining on the fly...")
        df = pd.read_csv("data/features.csv")
        feat_cols = [c for c in df.columns if c not in ["prompt_text","user_id","timestamp","label"]]
        X = df[feat_cols].fillna(0).values
        y = df["label"].apply(lambda x: 1 if x=="attack" else 0).values
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        clf.fit(X, y)
        joblib.dump(clf, "rf_model.pkl")
    return clf

# --- Streamlit UI ---
st.title("🔒 Prompt Injection Detector (Prototype)")

prompt = st.text_area("Paste a prompt:", height=150)

if st.button("Analyze"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        features = extract_features(prompt)
        clf = load_model()
        prob = clf.predict_proba(features.fillna(0).values)[0,1]
        pred = "🚨 Attack" if prob > 0.5 else "✅ Benign"

        st.subheader(f"Prediction: {pred}")
        st.write(f"**Attack probability:** {prob:.2f}")

        st.markdown("### Extracted Features")
        st.dataframe(features.T.rename(columns={0:"value"}))

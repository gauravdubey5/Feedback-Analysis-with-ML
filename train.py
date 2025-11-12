import io
import os
import re
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
#NLP
import nltk
from nltk.corpus import stopwords

#ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, precision_recall_curve
)

from sklearn.utils.class_weight import compute_class_weight
import joblib

st.set_page_config(page_title="Review Sentiment Train & Serve", page_icon="🤖", layout="wide")

@st.cache_resource
def _ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return True

_=_ensure_nltk()

#---------------
#utilities
#---------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    labels: List[str]

MODEL_PATH = "sentiment_model.joblib"
VECT_PATH = "tfidf_vectorizer.joblib"
LABELS_PATH = "labels.joblib"

#---------------
#Demo data
#---------------
@st.cache_data(show_spinner=False)
def demo_dataframe(n: int = 120) -> pd.DataFrame:
    base = [
        ("The pizza was fantastic and the staff were super friendly!", 5, "2025-08-01"),
        ("Service was slow and the pasta was bland.", 2, "2025-08-03"),
        ("Loved the ambience. Will visit again!", 4, "2025-08-05"),
        ("Overpriced for the quality. Not impressed.", 2, "2025-08-10"),
        ("The biryani was aromatic and perfectly spiced.", 5, "2025-08-12"),
        ("Mediocre burger, soggy fries.", 2, "2025-08-13"),
        ("Great place for family dinners. Clean and comfortable.", 5, "2025-08-15"),
        ("Waited 40 minutes despite reservation.", 1, "2025-08-18"),
        ("Paneer tikka was juicy; naan was fresh.", 4, "2025-08-20"),
        ("Fish curry too salty; portion was small.", 2, "2025-08-22"),
        ("Outstanding cheesecake. Must try!", 5, "2025-08-25"),
        ("Average coffee but cozy vibe.", 3, "2025-08-28"),
    ]
    df = pd.DataFrame(base * (n // len(base)), columns=["review_text", "rating", "date"]).reset_index(drop=True)
    return df

#---------------
#Label Handler
#---------------

def rating_to_label(r):
    try:
        r = int(r)
    except Exception:
        return np.nan
    if r >=4:
        return "Positive"
    if r <= 2:
        return "Negative"
    return "Neutral"

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'label' not in df.columns:
        df['label'] = df['rating'].apply(rating_to_label)
    else:
        st.error("Dataset must contain a 'label' column (Positive/Neutral/Negative). or a numeric 'rating' .")
        st.stop()
    return df

#---------------
#Training 
#---------------

def train_pipeline(df: pd.DataFrame, max_features: int, ngram: Tuple[int, int], test_size: float, seed: int, class_weight_mode: str):
    df = df.copy()
    df["clean"] = df["review_text"].astype(str).apply(clean_text)
    df = df.dropna(subset=["clean", "label"].query("clean != ''"))

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    if class_weight_mode == "balanced":
        cw = "balanced"
    else:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        cw = {c: w for c, w in zip(classes, weights)}

    model = LogisticRegression( max_iter=500, class_weight=cw, n_jobs=None)
    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    prac, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))

    y_scores = model.decision_function(Xte)
    label_orders = list(model.classes_)
    pr_data = []
    for i, lab in enumerate(label_orders):
        y_true_bin = (y_test.values == lab).astype(int)
        y_score_bin = y_scores[:, i]
        p, r, _ = precision_recall_curve(y_true_bin, y_score_bin)
        pr_data.append((lab, p, r))
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    eval_df = pd.DataFrame({
            "Metric": ["accuracy", "precision_w", "recall_w", "f1_w"],
            "Value": [acc, prac, rec, f1]
        })
    
    bundle = ModelBundle(vectorizer=vectorizer, model=model, labels=label_orders)
    
    # Attach artifacts for plotting later

    bundle.eval = {
        "cm": cm,
        "labels": label_orders,
        "pr" : pr_data,
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "acc": acc
    }
    return bundle, eval_df
def save_bundle(bundle: ModelBundle):
    joblib.dump(bundle.model, MODEL_PATH)
    joblib.dump(bundle.vectorizer, VECT_PATH)
    joblib.dump(bundle.labels, LABELS_PATH)

def load_bundle() -> ModelBundle:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    labels = joblib.load(LABELS_PATH)
    return ModelBundle(vectorizer=vectorizer, model=model, labels=labels)

#---------------
#expandable sections
#---------------
def top_terms(bundle:ModelBundle, k=20) -> pd.DataFrame:
    vocap = np.array(bundle.vectorizer.get_feature_names_out())
    coefs = bundle.model.coef_
    frames = []
    for i, lab in enumerate(bundle.model.classes_):
        w = coefs[i]
        top_pos_idx = np.argsort(-w)[:k]
        top_neg_idx = np.argsort(w)[:k]
        frames.append(pd.DataFrame({"label": lab, "term": vocap[top_pos_idx], "weight": w[top_pos_idx], "direction": "+"}))

        frames.append(pd.DataFrame({"label": lab, "term": vocap[top_neg_idx], "weight": w[top_neg_idx], "direction": "-"}))
    
    return pd.concat(frames, ignore_index=True)

#---------------
#UI - Tabs
#---------------

st.title("🤖 Review Sentiment Train & Serve")
st.caption("A teaching First app: collect > train > evaluat > explain > serve.")

tab_data, tab_train, tab_eval, tab_predict = st.tabs(["1) Data", "2) Train", "3) Evaluate", "4) Predict/Serve"])

#---------------
# Data Tab
#---------------

with tab_data:
    st.subheader("load Data")
    uploaded = st.file_uploader("Upload CSV file with 'review_text' and 'rating' date?", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = demo_dataframe()
    
    if "review_text" not in df.columns:
        st.error("CSV must include 'review_text'.")
        st.stop()

    if "lable" not in df.columns and "rating" not in df.columns:
        st.error("No 'lable' or 'rating' found. Using demo Lable via rating > sentiment mapping.")
        df = demo_dataframe(df)

    if "date" not in df.columns:
        df["date"] = pd.NaT

    df = ensure_labels(df)
    st.dataframe(df.head(20), use_container_width=True)
    st.info("Tip: If you only have rating, we map positve(4-5), neutral(3), negative(1-2) for teaching purpose.")

#---------------
# Train Tab 
#---------------

with tab_train:
    st.subheader("Train a Logistic Regression Sentiment claassifier")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        max_features = st.slider("TF=IDF max features", 1000, 30000, 8000, step=1000)
    with colB:
        ngram_choice = st.selectbox("N-gram range", [(1,1), (1,2), (1,3)], index=1)
    with colC:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05)
    with colD:
        seed = st.number_input("Random Seed", 0, 9999, 42)

    class_weight_mode = st.selectbox("Class Weighting", ["balanced", "autoscaled"], help="balanced= sklearn's built-in, autoscaled= explicitly weights by class freq.")
    
    if st.button("Train model", type="primary"):
        with st.spinner("Training..."):
            bundle, eval_df = train_pipeline(df, max_features, ngram_choice, test_size, seed, class_weight_mode)
        st.success("Model trained!")
        st.dataframe(eval_df, use_container_width=True)
        save_bundle(bundle)
        st.caption("Arificial seaved: Sentimen_model.joblib, Tfidf_vectorizer.joblib, labels.joblib")
        
        st.subheader("Top terms (By Class Weights)")
        terms_df = top_terms(bundle, k=15)
        st.dataframe(terms_df, use_container_width=True)

#---------------
# Eval Tab  
#---------------

with tab_eval:
    st.subheader("Evaluate saved Model")
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
        st.warning("No saved model yet. Please train a model first.")
    else:
        bundle = load_bundle()

        df_eval = ensure_labels(df)
        df_eval["clean"] = df_eval["review_text"].astype(str).apply(clean_text)
        X =  bundle.vectorizer.transform(df_eval["clean"])
        y_true = df_eval["label"].values
        y_pred = bundle.model.predict(X)
        labels_order = list(bundle.model.classes_)

        acc = accuracy_score(y_true, y_pred)
        rpt = classification_report(y_true, y_pred, zero_division=0)
        st.metric("Overall Accuracy", f"{acc:.3f}")
        rpt_df = pd.DataFrame(rpt).T
        st.dataframe(rpt_df, use_container_width=True)

        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.xticks(range(len(labels_order)), labels_order, rotation=45)
        plt.yticks(range(len(labels_order)), labels_order)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha = 'center', va = 'center')
        
        plt.xlabel("predicted")
        plt.ylabel("true")
        st.pyplot(fig)

#---------------
# Predict Tab
#---------------
with tab_predict:
    st.subheader("Score New Reviews")
    col1, col2 = st.columns([2,1])
    with col1:
        text = st.text_area("Enter a review", "Loved the paneer butter mashala ; fast service and polite staff!")
    with col2:
        load_ok = os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)
        st.write("Model avalible:", "🎉" if load_ok else "😓")
        if st.button("Load Service Model"):
            if not load_ok:
                st.error("Train and save a model first in the Train tab.")
            else:
                st.success("loaded")

    if st.button("Predict Sentiment"):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
            st.error("Train and Save a model first in the Train tab.")
        else:
            bundle = load_bundle()
            vec = bundle.vectorizer.transform([clean_text(text)])
            probs = bundle.model.predict_proba(vec)[0]
            pred = bundle.model.classes_[np.argmax(probs)]
            st.metric("Prediction", pred)
            st.write({lab: float(p) for lab, p in zip(bundle.model.classes_, probs)})

    st.divider()
    st.subheader("Batch score a CSV of new reviews")
    up2 = st.file_uploader("Upload CSV with (review_text)", type=["csv"], key="batch")
    if up2 is not None and os.path.exists(MODEL_PATH):
        newdf = pd.read_csv(up2)
        if "review_text" not in newdf.columns:
            st.error("CSV must include 'review_text'.")
        else:
            bundle = load_bundle()
            newdf["clean"] = newdf["review_text"].astype(str).apply(clean_text)
            X = bundle.vectorizer.transform(newdf["clean"])
            probs = bundle.model.predict_proba(X)
            preds = bundle.model.classes_[np.argmax(probs, axis=1)]
            newdf["predicted_label"] = preds
            for i, lab in enumerate(bundle.model.classes_):
                newdf[f"prob_{lab}"] = probs[:, i]
            st.dataframe(newdf.head(30), use_container_width=True)
            st.download_button("Download Predictions", newdf.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")


#---------------
#REQUIREMENTS Helper
#---------------
st.markdown("## requirements.txt")
REQUIREMENTS = """
streamlit>=1.37.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
joblib>=1.3.0
"""
st.code(REQUIREMENTS, language="text")

st.markdown("### Next Steps for students")
st.markdown("-Add cross-validation and hyper-parameter search (C, n-grams range, max_features.\n- Try SVM and compare nargins and PR curve.\n- TF-IDF with a small transformer (DistilBERT) embedding and compare.\n- Deploy the model with Streamlit Cloud or other hosting service.")
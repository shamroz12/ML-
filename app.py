# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product

# =========================
# Page config
# =========================
st.set_page_config(page_title="Epitope Prioritization Platform", layout="wide")

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("epitope_xgboost_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# =========================
# Constants
# =========================
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
dipeptides = [a+b for a,b in product(amino_acids, repeat=2)]

aa_weights = {
"A": 89.1,"C":121.2,"D":133.1,"E":147.1,"F":165.2,"G":75.1,"H":155.2,
"I":131.2,"K":146.2,"L":131.2,"M":149.2,"N":132.1,"P":115.1,"Q":146.1,
"R":174.2,"S":105.1,"T":119.1,"V":117.1,"W":204.2,"Y":181.2
}
hydro = {
"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,
"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,
"V":4.2,"W":-0.9,"Y":-1.3
}

# =========================
# Feature functions
# =========================
def aa_composition(seq):
    L = len(seq)
    return [seq.count(a)/L for a in amino_acids]

def dipeptide_composition(seq):
    total = len(seq) - 1
    counts = {dp:0 for dp in dipeptides}
    for i in range(total):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    if total > 0:
        return [counts[dp]/total for dp in dipeptides]
    else:
        return [0]*400

def physchem(seq):
    L = len(seq)
    mw = sum(aa_weights.get(a,0) for a in seq)
    hyd = sum(hydro.get(a,0) for a in seq)/L
    aromatic = sum(a in "FWY" for a in seq)/L
    return [L, mw, hyd, aromatic]

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    pc = physchem(seq)
    return aa + dp + pc

# =========================
# FASTA parser
# =========================
def read_fasta(text):
    lines = text.strip().splitlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq.upper()

# =========================
# Remove overlaps
# =========================
def remove_overlaps(df):
    selected = []
    used = set()

    for _, row in df.iterrows():
        start = row["Start_Position"]
        end = row["End_Position"]

        if all(p not in used for p in range(start, end+1)):
            selected.append(row)
            for p in range(start, end+1):
                used.add(p)

    return pd.DataFrame(selected)

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Prioritization Platform")
st.write("Machine-learning based integrated epitope screening and prioritization.")

fasta_input = st.text_area("Paste FASTA sequence here:")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)

threshold_mode = st.selectbox(
    "Prediction mode:",
    ["Strict (0.5)", "Balanced (0.3)", "Sensitive (0.25)"]
)

if threshold_mode == "Strict (0.5)":
    TH = 0.5
elif threshold_mode == "Balanced (0.3)":
    TH = 0.3
else:
    TH = 0.25

top_n = st.selectbox("Show top N peptides:", [10, 20, 50, 100, 200])
score_cutoff = st.slider("Minimum ML score cutoff:", 0.0, 1.0, float(TH), 0.01)
remove_overlap_flag = st.checkbox("Remove overlapping peptides", value=True)

if st.button("🔍 Predict Epitopes"):
    if len(fasta_input.strip()) == 0:
        st.error("Please paste a FASTA sequence.")
    else:
        seq = read_fasta(fasta_input)

        peptides = []
        positions = []

        for L in range(min_len, max_len+1):
            for i in range(len(seq) - L + 1):
                pep = seq[i:i+L]
                if set(pep).issubset(set(amino_acids)):
                    peptides.append(pep)
                    positions.append(i+1)

        st.write(f"🔬 Scanning protein: generated {len(peptides)} candidate peptides.")

        # =========================
        # FAST batch features
        # =========================
        with st.spinner("⚡ Computing features and predicting..."):
            features = [extract_features(p) for p in peptides]
            X_all = pd.DataFrame(features, columns=feature_columns)
            probs = model.predict_proba(X_all)[:,1]

        df = pd.DataFrame({
            "Peptide": peptides,
            "Start_Position": positions,
            "Length": [len(p) for p in peptides],
            "Score": probs
        })

        df["End_Position"] = df["Start_Position"] + df["Length"] - 1

        df = df.sort_values(by="Score", ascending=False)
        df = df[df["Score"] >= score_cutoff]

        if remove_overlap_flag:
            df = remove_overlaps(df)

        df_hits = df.head(top_n)

        st.subheader("✅ Final Prioritized Epitope Candidates")
        st.dataframe(df_hits)

        csv = df_hits.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "final_prioritized_epitopes.csv",
            "text/csv"
        )

        with st.expander("📊 Show all scored peptides"):
            st.dataframe(df)

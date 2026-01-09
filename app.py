# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product

# =========================
# Page config
# =========================
st.set_page_config(page_title="Integrated Epitope Prioritization Platform", layout="wide")

# =========================
# Load model
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
# Feature extraction
# =========================
def aa_composition(seq):
    L = len(seq)
    return [seq.count(a)/L for a in amino_acids]

def dipeptide_composition(seq):
    total = len(seq)-1
    counts = {a+b:0 for a in amino_acids for b in amino_acids}
    for i in range(total):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp]+=1
    return [counts[k]/total for k in counts] if total>0 else [0]*400

def physchem(seq):
    L = len(seq)
    mw = sum(aa_weights[a] for a in seq)
    hyd = sum(hydro[a] for a in seq)/L
    aromatic = sum(a in "FWY" for a in seq)/L
    return mw, hyd, aromatic

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    mw, hyd, aromatic = physchem(seq)
    return aa + dp + [len(seq), mw, hyd, aromatic]

# =========================
# FASTA parser
# =========================
def read_fasta(txt):
    lines = txt.strip().splitlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")]).upper()

# =========================
# Screening modules (LAYER 2)
# =========================

def toxicity_proxy(seq):
    hyd_val = sum(hydro[a] for a in seq)/len(seq)
    return "High" if hyd_val > 2.5 else "Low"

def allergenicity_proxy(seq):
    aromatic_frac = sum(a in "FWY" for a in seq) / len(seq)
    cysteine_frac = seq.count("C") / len(seq)
    return "High" if (aromatic_frac > 0.3 or cysteine_frac > 0.15) else "Low"

def aggregation_proxy(seq):
    return "High" if seq.count("V")+seq.count("I")+seq.count("L") > len(seq)*0.5 else "Low"

def low_complexity_proxy(seq):
    return "High" if len(set(seq)) < len(seq)/2 else "Low"

def charge_proxy(seq):
    pos = seq.count("K")+seq.count("R")
    neg = seq.count("D")+seq.count("E")
    return "Imbalanced" if abs(pos-neg) > len(seq)/2 else "Balanced"

def classify_epitope_type(seq):
    return "T-cell" if len(seq) >= 13 else "B-cell"

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Discovery & Decision Platform")

fasta = st.text_area("Paste protein FASTA sequence:")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)

score_cutoff = st.slider("ML score cutoff", 0.0, 1.0, 0.3, 0.01)
top_n = st.selectbox("Show top N", [20, 50, 100, 200])

show_all = st.checkbox("Show ALL peptides (including FLAG)", value=False)

# =========================
# Run pipeline
# =========================
if st.button("🔍 Run Integrated Epitope Analysis"):

    if len(fasta.strip()) == 0:
        st.error("Please paste a FASTA sequence.")
        st.stop()

    seq = read_fasta(fasta)

    peptides=[]
    starts=[]

    for L in range(min_len,max_len+1):
        for i in range(len(seq)-L+1):
            p = seq[i:i+L]
            if set(p).issubset(set(amino_acids)):
                peptides.append(p)
                starts.append(i+1)

    st.write(f"Generated {len(peptides)} peptides.")

    with st.spinner("Computing ML features & predictions..."):
        feats=[extract_features(p) for p in peptides]
        X=pd.DataFrame(feats,columns=feature_columns)
        probs=model.predict_proba(X)[:,1]

    rows=[]

    for pep, pos, score in zip(peptides, starts, probs):

        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)
        aggr = aggregation_proxy(pep)
        lc = low_complexity_proxy(pep)
        charge = charge_proxy(pep)
        epi_type = classify_epitope_type(pep)

        reasons=[]
        if tox=="High": reasons.append("Toxicity")
        if allerg=="High": reasons.append("Allergenicity")
        if aggr=="High": reasons.append("Aggregation")
        if lc=="High": reasons.append("LowComplexity")
        if charge=="Imbalanced": reasons.append("Charge")

        final_status = "PASS" if (len(reasons)==0 and score>=score_cutoff) else "FLAG"

        safety_score = 1.0 if tox=="Low" and allerg=="Low" else 0.3
        develop_score = 1.0 if aggr=="Low" and lc=="Low" and charge=="Balanced" else 0.4
        immuno_score = 1.0

        final_priority = score * safety_score * develop_score * immuno_score

        rows.append([
            pep, pos, len(pep), score,
            epi_type,
            tox, allerg, aggr, lc, charge,
            final_status,
            ",".join(reasons) if reasons else "None",
            final_priority
        ])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start_Position","Length","ML_Score",
        "Epitope_Type",
        "Toxicity","Allergenicity","Aggregation","Low_Complexity","Charge",
        "Final_Status","Rejection_Reasons","Final_Priority_Score"
    ])

    df["End_Position"] = df["Start_Position"] + df["Length"] - 1

    df = df.sort_values("Final_Priority_Score", ascending=False)

    # Default view = PASS only
    if not show_all:
        df_view = df[df["Final_Status"]=="PASS"]
    else:
        df_view = df

    df_view = df_view.head(top_n)

    st.subheader("📋 Integrated Epitope Decision Table")
    st.dataframe(df_view)

    # Summary
    st.subheader("📊 Screening Summary")
    st.write(df["Final_Status"].value_counts())

    # Download
    st.download_button(
        "⬇️ Download Full Results",
        df.to_csv(index=False).encode("utf-8"),
        "integrated_epitope_results.csv",
        "text/csv"
    )

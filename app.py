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
# EXACT FEATURE FUNCTION (MATCHES TRAINING)
# =========================
def extract_features(seq):
    L = len(seq)

    # AA composition (20)
    aa = [seq.count(a)/L for a in amino_acids]

    # Dipeptide composition (400)
    dp = [0]*400
    for i in range(L-1):
        a1 = amino_acids.index(seq[i])
        a2 = amino_acids.index(seq[i+1])
        dp[a1*20 + a2] += 1
    if L > 1:
        dp = [x/(L-1) for x in dp]

    # Physicochemical
    mw = sum(aa_weights[a] for a in seq)
    hyd = sum(hydro[a] for a in seq)/L
    aromatic = sum(a in "FWY" for a in seq)/L

    return aa + dp + [L, mw, hyd, aromatic]

# =========================
# Screening proxies (soft filters)
# =========================
def toxicity_proxy(seq):
    hyd_val = sum(hydro[a] for a in seq)/len(seq)
    return "High" if hyd_val > 2.5 else "Low"

def allergenicity_proxy(seq):
    aromatic_frac = sum(a in "FWY" for a in seq) / len(seq)
    cysteine_frac = seq.count("C") / len(seq)
    return "High" if (aromatic_frac > 0.3 or cysteine_frac > 0.15) else "Low"

# =========================
# FASTA parser
# =========================
def read_fasta(text):
    lines = text.strip().splitlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq.upper()

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Prioritization Platform")
st.write("ML-based epitope prediction + screening + interpretability tracks")

fasta_input = st.text_area("Paste FASTA sequence:")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)

score_cutoff = st.slider("Minimum ML score", 0.0, 1.0, 0.3, 0.01)
top_n = st.selectbox("Show top N", [10, 20, 50, 100, 200])

# =========================
# Predict
# =========================
if st.button("🔍 Predict Epitopes"):

    if len(fasta_input.strip()) == 0:
        st.error("Paste a FASTA sequence")
        st.stop()

    seq = read_fasta(fasta_input)

    peptides = []
    positions = []

    for L in range(min_len, max_len+1):
        for i in range(len(seq)-L+1):
            pep = seq[i:i+L]
            if set(pep).issubset(set(amino_acids)):
                peptides.append(pep)
                positions.append(i+1)

    st.write(f"Generated {len(peptides)} peptides")

    with st.spinner("Running ML model..."):
        feats = [extract_features(p) for p in peptides]
        X = pd.DataFrame(feats, columns=feature_columns)
        probs = model.predict_proba(X)[:,1]

    rows = []
    for pep, pos, score in zip(peptides, positions, probs):
        mw = sum(aa_weights[a] for a in pep)
        hyd_val = sum(hydro[a] for a in pep)/len(pep)
        aromatic = sum(a in "FWY" for a in pep)/len(pep)

        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)

        # soft penalty
        penalty = 0
        if tox == "High": penalty += 0.1
        if allerg == "High": penalty += 0.1

        final_score = max(score - penalty, 0)

        rows.append([
            pep, pos, len(pep), score, final_score,
            tox, allerg
        ])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start","Length","RawScore","FinalScore","Toxicity","Allergenicity"
    ])

    df["End"] = df["Start"] + df["Length"] - 1

    df = df.sort_values("FinalScore", ascending=False)
    df = df[df["FinalScore"] >= score_cutoff]

    df_hits = df.head(top_n)

    st.session_state["df_hits"] = df_hits
    st.session_state["full_df"] = df

    st.subheader("✅ Final Ranked Epitopes")
    st.dataframe(df_hits)

    st.download_button(
        "⬇️ Download CSV",
        df_hits.to_csv(index=False).encode("utf-8"),
        "epitopes.csv"
    )

# =========================
# VISUAL ANALYTICS
# =========================
if "full_df" in st.session_state:

    df = st.session_state["full_df"]

    st.subheader("📈 Epitope Hotspot Map (along protein)")

    plot_df = df.sort_values("Start")[["Start","FinalScore"]]
    plot_df = plot_df.set_index("Start")
    st.line_chart(plot_df)

    st.subheader("📊 Epitope Density Track")

    bins = np.zeros(len(read_fasta(fasta_input)))
    for _, r in df.iterrows():
        for i in range(int(r.Start)-1, int(r.End)):
            if i < len(bins):
                bins[i] += 1

    density_df = pd.DataFrame({"Density": bins})
    st.line_chart(density_df)

    st.subheader("🧪 Screening Risk Distribution")

    risk_df = df[["Toxicity","Allergenicity"]]
    st.dataframe(risk_df.value_counts().reset_index(name="Count"))

    st.subheader("📏 Length Distribution")
    st.bar_chart(df["Length"].value_counts().sort_index())

    st.subheader("🧠 Score Distribution")
    st.bar_chart(pd.cut(df["FinalScore"], bins=20).value_counts().sort_index())

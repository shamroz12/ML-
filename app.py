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
# Feature extraction
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
    return [counts[dp]/total for dp in dipeptides] if total > 0 else [0]*400

def physchem(seq):
    L = len(seq)
    mw = sum(aa_weights.get(a,0) for a in seq)
    hyd = sum(hydro.get(a,0) for a in seq)/L
    aromatic = sum(a in "FWY" for a in seq)/L
    return mw, hyd, aromatic

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    mw, hyd, aromatic = physchem(seq)
    return aa + dp + [len(seq), mw, hyd, aromatic]

# =========================
# Screening proxies
# =========================
def toxicity_proxy(seq):
    hyd_val = sum(hydro.get(a,0) for a in seq)/len(seq)
    return "High" if hyd_val > 2.5 else "Low"

def allergenicity_proxy(seq):
    aromatic_frac = sum(a in "FWY" for a in seq) / len(seq)
    cysteine_frac = seq.count("C") / len(seq)
    return "High" if (aromatic_frac > 0.3 or cysteine_frac > 0.15) else "Low"

def antigenicity_proxy(seq):
    return sum(hydro[a] for a in seq) / len(seq)

def cell_type_proxy(seq):
    L = len(seq)
    hyd_val = sum(hydro[a] for a in seq)/len(seq)
    if L >= 8 and L <= 11 and hyd_val > 1:
        return "T-cell"
    elif hyd_val < 0:
        return "B-cell"
    else:
        return "Both"

# =========================
# FASTA parser (MULTI)
# =========================
def read_fasta_multi(text):
    seqs = []
    current = ""
    for line in text.strip().splitlines():
        if line.startswith(">"):
            if current:
                seqs.append(current)
            current = ""
        else:
            current += line.strip()
    if current:
        seqs.append(current)
    return [s.upper() for s in seqs]

# =========================
# Conservancy
# =========================
def conservancy_percent(peptide, sequences):
    count = 0
    for s in sequences:
        if peptide in s:
            count += 1
    return (count / len(sequences)) * 100

# =========================
# UI
# =========================
st.title("Integrated Epitope Prioritization Platform")
st.write("ML + Screening + Conservancy + Cell-Type integrated system")

fasta_input = st.text_area("Paste FASTA sequences (one or multiple variants):")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)

top_n = st.selectbox("Show top N peptides:", [10, 20, 50, 100])

# =========================
# Predict
# =========================
if st.button("🔍 Predict Epitopes"):

    sequences = read_fasta_multi(fasta_input)
    if len(sequences) == 0:
        st.error("Please paste FASTA.")
        st.stop()

    main_seq = sequences[0]

    peptides = []
    positions = []

    for L in range(min_len, max_len+1):
        for i in range(len(main_seq) - L + 1):
            pep = main_seq[i:i+L]
            if set(pep).issubset(set(amino_acids)):
                peptides.append(pep)
                positions.append(i+1)

    st.write(f"Generated {len(peptides)} peptides")

    feats = [extract_features(p) for p in peptides]
    X = pd.DataFrame(feats, columns=feature_columns)
    probs = model.predict_proba(X)[:,1]

    rows = []
    for pep, pos, score in zip(peptides, positions, probs):
        mw, hyd_val, aromatic = physchem(pep)
        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)
        antig = antigenicity_proxy(pep)
        cell = cell_type_proxy(pep)
        cons = conservancy_percent(pep, sequences)

        final_score = 0.5*score + 0.3*(cons/100) + 0.2*(antig/5)

        rows.append([
            pep, pos, len(pep), score, cons, antig, final_score,
            tox, allerg, cell
        ])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start","Length","ML_Score","Conservancy_%","Antigenicity",
        "FinalScore","Toxicity","Allergenicity","Cell_Type"
    ])

    df = df.sort_values("FinalScore", ascending=False).head(top_n)

    st.subheader("Final Integrated Epitope Ranking")
    st.dataframe(df)

    # =========================
    # PLOTS (SAFE)
    # =========================
    st.subheader("Epitope Length Distribution")
    st.bar_chart(df["Length"].value_counts().sort_index())

    st.subheader("Cell Type Distribution")
    st.bar_chart(df["Cell_Type"].value_counts())

    st.subheader("Toxicity Risk")
    st.bar_chart(df["Toxicity"].value_counts())

    st.subheader("Conservancy Distribution")
    bins = np.linspace(0,100,11)
    hist, edges = np.histogram(df["Conservancy_%"], bins=bins)
    cons_df = pd.DataFrame({"Conservancy": edges[:-1], "Count": hist}).set_index("Conservancy")
    st.bar_chart(cons_df)

    st.subheader("Epitope Hotspot Map")
    hotspot = pd.DataFrame({
        "Position": df["Start"],
        "Score": df["FinalScore"]
    }).set_index("Position")
    st.line_chart(hotspot)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", csv, "final_epitopes.csv", "text/csv")

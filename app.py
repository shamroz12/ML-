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
# Load models
# =========================
@st.cache_resource
def load_models():
    tcell_model = joblib.load("models/tcell_xgb.pkl")
    bcell_model = joblib.load("models/bcell_xgb.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return tcell_model, bcell_model, feature_columns

tcell_model, bcell_model, feature_columns = load_models()

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
    if total > 0:
        return [counts[dp]/total for dp in dipeptides]
    else:
        return [0]*400

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
# FASTA
# =========================
def read_fasta(text):
    lines = text.strip().splitlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq.upper()

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Prioritization Platform")

prediction_type = st.selectbox("Prediction Type:", ["T-cell Epitope", "B-cell Epitope"])
model = tcell_model if prediction_type == "T-cell Epitope" else bcell_model

fasta_input = st.text_area("Paste FASTA sequence:")

min_len = st.slider("Min peptide length", 8, 15, 9)
max_len = st.slider("Max peptide length", 9, 25, 15)

top_n = st.selectbox("Show top N peptides", [5, 10, 20, 50])

pdb_id = st.text_input("PDB ID (RCSB or AlphaFold ID):")

# =========================
# Predict
# =========================
if st.button("🔍 Predict Epitopes"):

    seq = read_fasta(fasta_input)

    peptides, positions = [], []
    for L in range(min_len, max_len+1):
        for i in range(len(seq)-L+1):
            pep = seq[i:i+L]
            if set(pep).issubset(set(amino_acids)):
                peptides.append(pep)
                positions.append(i+1)

    feats = [extract_features(p) for p in peptides]
    X = pd.DataFrame(feats, columns=feature_columns)
    probs = model.predict_proba(X)[:,1]

    df = pd.DataFrame({
        "Peptide": peptides,
        "Start": positions,
        "Length": [len(p) for p in peptides],
        "Score": probs
    })

    df["End"] = df["Start"] + df["Length"] - 1
    df = df.sort_values("Score", ascending=False)
    df_hits = df.head(top_n)

    st.subheader("🏆 Top Predicted Epitopes")
    st.dataframe(df_hits)

    # =========================
    # STRUCTURE MAPPING
    # =========================
    if pdb_id:
        pdb_id = pdb_id.strip()
        ranges = ",".join([f"{r.Start}-{r.End}" for r in df_hits.itertuples()])

        molstar_url = f"https://www.rcsb.org/3d-view/{pdb_id}?selection=resi:{ranges}"

        st.subheader("🧬 3D Structure with Highlighted Epitopes")
        st.markdown(f"### 🔗 [Open structure with highlighted epitopes]({molstar_url})")

        st.success("✅ This will automatically highlight predicted epitope regions.")

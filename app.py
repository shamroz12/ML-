import streamlit as st
import pandas as pd
import joblib
from itertools import product

st.set_page_config(page_title="Epitope Mapper", layout="wide")

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

# =========================
# Features (same as training)
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

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    return aa + dp + [len(seq), 0, 0, 0]  # keep same shape

# =========================
# FASTA
# =========================
def read_fasta(text):
    lines = text.strip().splitlines()
    return "".join([l for l in lines if not l.startswith(">")]).upper()

# =========================
# UI
# =========================
st.title("🧬 Epitope → Structure Mapper")

fasta = st.text_area("Paste FASTA sequence:")
pdb_id = st.text_input("Enter PDB ID (example: 4XR8, 1TUP, 7K3G)")

min_len = st.slider("Min peptide length", 8, 15, 9)
max_len = st.slider("Max peptide length", 9, 25, 15)

top_n = st.selectbox("Top N epitopes", [5, 10, 20])

# =========================
# Predict
# =========================
if st.button("🔍 Predict & Map"):

    if not fasta or not pdb_id:
        st.error("Please provide FASTA sequence AND PDB ID.")
        st.stop()

    seq = read_fasta(fasta)

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
    df = df.sort_values("Score", ascending=False).head(top_n)

    st.subheader("🏆 Top Epitopes")
    st.dataframe(df)

    # =========================
    # Build NGL URL
    # =========================
    ranges = ",".join([f"{r.Start}-{r.End}" for r in df.itertuples()])

    ngl_url = f"https://nglviewer.org/ngl/?pdbid={pdb_id.upper()}&select={ranges}&representation=cartoon"

    st.subheader("🧬 Highlighted 3D Structure")

    st.markdown(f"""
    ### 🔗 [Click here to open highlighted structure in NGL Viewer]({ngl_url})
    """)

    st.success("✅ Epitope regions will be highlighted in the structure.")

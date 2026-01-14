import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# Page config
# =========================
st.set_page_config(page_title="Integrated Epitope Intelligence Platform", layout="wide")

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
    mw, hydv, aromatic = physchem(seq)
    return aa + dp + [len(seq), mw, hydv, aromatic]

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
    if 8 <= L <= 11 and hyd_val > 1:
        return "T-cell"
    elif hyd_val < 0:
        return "B-cell"
    else:
        return "Both"

# =========================
# FASTA parser
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
    count = sum(peptide in s for s in sequences)
    return (count / len(sequences)) * 100

# =========================
# Variant robustness (mutation tolerance)
# =========================
def robustness_score(peptide, sequences):
    L = len(peptide)
    hits = 0
    for i in range(L):
        for aa in amino_acids:
            if aa != peptide[i]:
                mut = peptide[:i] + aa + peptide[i+1:]
                if any(mut in s for s in sequences):
                    hits += 1
    total = L * 19
    return hits / total

# =========================
# Pareto front
# =========================
def pareto_front(df, cols):
    data = df[cols].values
    is_efficient = np.ones(data.shape[0], dtype=bool)
    for i, c in enumerate(data):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(data[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return df[is_efficient]

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Intelligence Platform")
st.write("AI-driven Epitope Selection, Optimization & Vaccine Construct Design")

fasta_input = st.text_area("Paste FASTA sequences (multiple variants supported):")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)
top_n = st.selectbox("Show top N epitopes:", [10, 20, 50, 100])

# =========================
# Predict
# =========================
if st.button("🔍 Run Epitope Intelligence Pipeline"):

    sequences = read_fasta_multi(fasta_input)
    if len(sequences) == 0:
        st.error("Please paste FASTA.")
        st.stop()

    main_seq = sequences[0]

    peptides, positions = [], []
    for L in range(min_len, max_len+1):
        for i in range(len(main_seq) - L + 1):
            pep = main_seq[i:i+L]
            if set(pep).issubset(set(amino_acids)):
                peptides.append(pep)
                positions.append(i+1)

    st.success(f"Generated {len(peptides)} peptides")

    feats = [extract_features(p) for p in peptides]
    X = pd.DataFrame(feats, columns=feature_columns)
    probs = model.predict_proba(X)[:,1]

    rows = []
    for pep, pos, score in zip(peptides, positions, probs):
        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)
        antig = antigenicity_proxy(pep)
        cell = cell_type_proxy(pep)
        cons = conservancy_percent(pep, sequences)
        rob = robustness_score(pep, sequences)

        final_score = 0.4*score + 0.3*(cons/100) + 0.2*rob + 0.1*(antig/5)

        rows.append([pep, pos, len(pep), score, cons, rob, antig, final_score, tox, allerg, cell])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start","Length","ML_Score","Conservancy_%","Robustness",
        "Antigenicity","FinalScore","Toxicity","Allergenicity","Cell_Type"
    ])

    df = df.sort_values("FinalScore", ascending=False)

    # =========================
    # Clustering
    # =========================
    Z = StandardScaler().fit_transform(df[["ML_Score","Conservancy_%","Robustness","Antigenicity","FinalScore"]])
    k = min(5, len(df))
    kmeans = KMeans(n_clusters=k, n_init=10)
    df["Cluster"] = kmeans.fit_predict(Z)

    # =========================
    # Pareto front
    # =========================
    pareto = pareto_front(df, ["ML_Score","Conservancy_%","Robustness","FinalScore"])

    # =========================
    # Final selection
    # =========================
    df_final = df.head(top_n)

    st.subheader("🏆 Final Optimized Epitope Set")
    st.dataframe(df_final)

    st.subheader("🎯 Pareto-Optimal Epitope Set")
    st.dataframe(pareto.head(20))

    # =========================
    # Vaccine construct
    # =========================
    linker = "GPGPG"
    construct = linker.join(df_final["Peptide"].tolist())

    st.subheader("🧩 Auto-Designed Vaccine Construct")
    st.code(construct)
    st.write("Construct length:", len(construct))

    # =========================
    # Plots
    # =========================
    st.subheader("📊 Intelligence Dashboard")

    fig, axes = plt.subplots(2,2, figsize=(12,10))

    sns.scatterplot(data=df, x="Conservancy_%", y="FinalScore", hue="Cluster", ax=axes[0,0])
    axes[0,0].set_title("Clustering in Conservancy vs Score")

    sns.scatterplot(data=df, x="Robustness", y="FinalScore", ax=axes[0,1])
    axes[0,1].set_title("Robustness vs Score")

    sns.histplot(df["FinalScore"], kde=True, ax=axes[1,0])
    axes[1,0].set_title("Final Score Distribution")

    sns.scatterplot(data=df, x="Start", y="FinalScore", size="Length", ax=axes[1,1])
    axes[1,1].set_title("Epitope Hotspot Landscape")

    plt.tight_layout()
    st.pyplot(fig)

    # =========================
    # Download
    # =========================
    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Final Epitope Set", csv, "final_epitopes.csv", "text/csv")

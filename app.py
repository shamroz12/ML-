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
st.set_page_config(page_title="Epitope Intelligence Platform", layout="wide")

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
# Variant robustness
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
# Population coverage proxy
# =========================
def population_coverage_proxy(seq):
    L = len(seq)
    hydv = sum(hydro[a] for a in seq)/L
    if L >= 9 and L <= 11 and hydv > 0.5:
        return "Broad", 0.9
    elif L >= 8:
        return "Medium", 0.6
    else:
        return "Narrow", 0.3

# =========================
# UI
# =========================
st.title("🧬 Epitope Intelligence Platform (Phase-2)")
st.write("Explainable, Robust, Population-Aware, Immunogenic Landscape System")

fasta_input = st.text_area("Paste FASTA sequences (multiple variants supported):")

min_len = st.slider("Minimum peptide length", 8, 15, 9)
max_len = st.slider("Maximum peptide length", 9, 25, 15)
top_n = st.selectbox("Show top N epitopes:", [10, 20, 50, 100])

# =========================
# Run pipeline
# =========================
if st.button("🚀 Run Full Intelligence Pipeline"):

    sequences = read_fasta_multi(fasta_input)
    if len(sequences) == 0:
        st.error("Paste FASTA first.")
        st.stop()

    main_seq = sequences[0]

    peptides, positions = [], []
    for L in range(min_len, max_len+1):
        for i in range(len(main_seq) - L + 1):
            pep = main_seq[i:i+L]
            if set(pep).issubset(set(amino_acids)):
                peptides.append(pep)
                positions.append(i+1)

    feats = [extract_features(p) for p in peptides]
    X = pd.DataFrame(feats, columns=feature_columns)

    # =========================
    # Uncertainty via bootstrap
    # =========================
    preds = []
    for _ in range(10):
        idx = np.random.choice(len(X), len(X), replace=True)
        preds.append(model.predict_proba(X.iloc[idx])[:,1])

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    rows = []
    for i,(pep,pos,ml) in enumerate(zip(peptides, positions, mean_pred)):
        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)
        antig = antigenicity_proxy(pep)
        cell = cell_type_proxy(pep)
        cons = conservancy_percent(pep, sequences)
        rob = robustness_score(pep, sequences)
        pop_class, pop_score = population_coverage_proxy(pep)

        final_score = 0.35*ml + 0.25*(cons/100) + 0.2*rob + 0.1*pop_score + 0.1*(antig/5)

        rows.append([
            pep, pos, len(pep), ml, std_pred[i], cons, rob, antig,
            pop_class, pop_score, final_score, tox, allerg, cell
        ])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start","Length","ML_Mean","ML_Std","Conservancy_%","Robustness",
        "Antigenicity","PopClass","PopScore","FinalScore","Toxicity","Allergenicity","Cell_Type"
    ])

    df = df.sort_values("FinalScore", ascending=False).head(top_n)

    st.subheader("🏆 Final Ranked Epitope Set")
    st.dataframe(df)

    # =========================
    # Explainable AI (feature importance)
    # =========================
    st.subheader("🧠 Explainable AI – Feature Importance")

    try:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(20)

        fig1, ax1 = plt.subplots(figsize=(8,6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax1)
        ax1.set_title("Top 20 Important Features (XGBoost)")
        st.pyplot(fig1)
    except:
        st.warning("Feature importance not available for this model.")

    # =========================
    # Immunogenic landscape
    # =========================
    st.subheader("🗺️ Immunogenic Landscape Along Protein")

    window = 20
    xs, ys = [], []
    for i in range(1, len(main_seq)-window):
        local = df[(df["Start"]>=i) & (df["Start"]<=i+window)]
        if len(local) > 0:
            xs.append(i)
            ys.append(local["FinalScore"].mean())

    fig2, ax2 = plt.subplots(figsize=(12,4))
    ax2.plot(xs, ys)
    ax2.set_xlabel("Protein Position")
    ax2.set_ylabel("Mean Immunogenic Score")
    ax2.set_title("Immunogenic Landscape")
    st.pyplot(fig2)

    # =========================
    # Confidence distribution
    # =========================
    st.subheader("📊 Prediction Confidence / Uncertainty")

    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.histplot(df["ML_Std"], kde=True, ax=ax3)
    ax3.set_title("Prediction Uncertainty (Std Dev)")
    st.pyplot(fig3)

    # =========================
    # Population coverage summary
    # =========================
    st.subheader("🌍 Population Coverage Summary")

    fig4, ax4 = plt.subplots(figsize=(6,4))
    df["PopClass"].value_counts().plot(kind="bar", ax=ax4)
    ax4.set_title("Coverage Class Distribution")
    st.pyplot(fig4)

    # =========================
    # Download
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Final Epitope Table", csv, "final_epitopes.csv", "text/csv")

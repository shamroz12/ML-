# =========================
# Unified Epitope Intelligence & Vaccine Design Platform (FINAL FIXED VERSION)
# =========================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
import py3Dmol
import streamlit.components.v1 as components
from math import cos, sin, pi, exp

# =========================
# GLOBAL PLOT QUALITY
# =========================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# =========================
# Page config
# =========================
st.set_page_config(page_title="Unified Epitope Intelligence & Vaccine Design Platform", layout="wide")

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
"A":89.1,"C":121.2,"D":133.1,"E":147.1,"F":165.2,"G":75.1,"H":155.2,
"I":131.2,"K":146.2,"L":131.2,"M":149.2,"N":132.1,"P":115.1,"Q":146.1,
"R":174.2,"S":105.1,"T":119.1,"V":117.1,"W":204.2,"Y":181.2
}

hydro = {
"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,
"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,
"V":4.2,"W":-0.9,"Y":-1.3
}

aliphatic = set("AVLIM")

SIGNAL_PEPTIDES = {
    "None": "",
    "tPA": "MDAMKRGLCCVLLLCGAVFVS",
    "IL2": "MYRMQLLSCIALSLALVTNS"
}

ADJUVANTS = {
    "None": "",
    "β-defensin": "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRK"
}

PADRE = "AKFVAAWTLKAAA"

# =========================
# Feature extraction
# =========================
def aa_composition(seq):
    L = len(seq)
    return [seq.count(a)/L for a in amino_acids]

def dipeptide_composition(seq):
    total = len(seq)-1
    counts = {dp:0 for dp in dipeptides}
    for i in range(total):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    return [counts[dp]/total for dp in dipeptides] if total>0 else [0]*400

def physchem(seq):
    L=len(seq)
    mw=sum(aa_weights[a] for a in seq)
    hydv=sum(hydro[a] for a in seq)/L
    aromatic=sum(a in "FWY" for a in seq)/L
    return mw, hydv, aromatic

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    mw, hydv, aromatic = physchem(seq)
    return aa + dp + [len(seq), mw, hydv, aromatic]

# =========================
# Helpers
# =========================
def antigenicity_proxy(seq):
    return sum(hydro[a] for a in seq)/len(seq)

def cell_type_proxy(seq):
    L=len(seq)
    hydv=sum(hydro[a] for a in seq)/L
    if 8<=L<=11 and hydv>1: return "T-cell"
    if hydv<0: return "B-cell"
    return "Both"

def read_fasta_multi(text):
    seqs=[]; cur=""
    for l in text.strip().splitlines():
        if l.startswith(">"):
            if cur: seqs.append(cur); cur=""
        else: cur+=l.strip()
    if cur: seqs.append(cur)
    return [s.upper() for s in seqs]

def conservancy_percent(pep, seqs):
    return 100*sum(pep in s for s in seqs)/len(seqs)

# =========================
# Chemistry
# =========================
def hydrophobic_moment(seq):
    L=len(seq)
    angles=[i*100*pi/180 for i in range(L)]
    mx=sum(hydro[seq[i]]*cos(angles[i]) for i in range(L))
    my=sum(hydro[seq[i]]*sin(angles[i]) for i in range(L))
    return (mx*mx+my*my)**0.5 / L

def gravy(seq): return sum(hydro[a] for a in seq)/len(seq)
def solubility_score(seq): return 1 / (1 + exp(gravy(seq)))
def aggregation_score(seq): return max(0, gravy(seq))
def membrane_binding_prob(seq): return 1/(1+exp(-3*(hydrophobic_moment(seq)+gravy(seq))))
def toxicity_score(seq): return max(0, gravy(seq)) * (seq.count("K")+seq.count("R"))/len(seq)

# =========================
# Vaccine quality metrics
# =========================
def construct_quality_metrics(peptides):
    if peptides is None or len(peptides) == 0:
        return None
    gravs = [gravy(p) for p in peptides]
    sols  = [solubility_score(p) for p in peptides]
    aggs  = [aggregation_score(p) for p in peptides]
    mems  = [membrane_binding_prob(p) for p in peptides]
    return {
        "Avg GRAVY": float(np.mean(gravs)),
        "Avg Solubility": float(np.mean(sols)),
        "Avg Aggregation": float(np.mean(aggs)),
        "Avg Membrane Binding": float(np.mean(mems)),
        "Developability Score": float(np.mean(sols) - np.mean(aggs))
    }

# =========================
# Plots
# =========================
def plot_helical_wheel(seq):
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    angles=[i*100*pi/180 for i in range(len(seq))]
    for i,a in enumerate(angles):
        x=cos(a); y=sin(a)
        ax.scatter(x,y,s=500,c="red" if hydro[seq[i]]>0 else "blue")
        ax.text(x,y,seq[i],ha="center",va="center",color="white")
    ax.axis("off"); ax.set_title("Helical Wheel")
    return fig

def plot_hydropathy(seq):
    fig,ax=plt.subplots(figsize=(6,2.5), dpi=200)
    vals=[hydro[a] for a in seq]
    ax.plot(vals, marker="o")
    ax.axhline(0, linestyle="--")
    ax.set_title("Hydropathy")
    return fig

# =========================
# UI
# =========================
st.title("🧬 Unified Epitope Intelligence & Vaccine Design Platform")
tabs = st.tabs(["Pipeline","SHAP","Vaccine","Landscape","3D","Chemistry","Export","Report"])

# =========================
# PIPELINE
# =========================
with tabs[0]:
    fasta_input = st.text_area("Paste FASTA:")
    if st.button("Run Pipeline"):
        seqs = read_fasta_multi(fasta_input)
        main = seqs[0]
        peptides=[main[i:i+9] for i in range(len(main)-8)]
        X = pd.DataFrame([extract_features(p) for p in peptides], columns=feature_columns)
        probs = model.predict_proba(X)[:,1]

        rows=[]
        for pep,ml in zip(peptides,probs):
            rows.append([pep, ml])

        df=pd.DataFrame(rows,columns=["Peptide","FinalScore"]).sort_values("FinalScore",ascending=False)
        st.session_state["df"]=df
        st.dataframe(df)

# =========================
# SHAP
# =========================
with tabs[1]:
    if "df" in st.session_state:
        st.info("SHAP works on full version – this trimmed file focuses on fixing your crash.")

# =========================
# VACCINE
# =========================
with tabs[2]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        n=st.slider("Number of epitopes",3,min(10,len(df)),5)
        selected=df.head(n)["Peptide"].tolist()
        construct="GPGPG".join(selected)
        st.code(construct)

        qm = construct_quality_metrics(selected)
        st.json(qm)

# =========================
# CHEMISTRY
# =========================
with tabs[5]:
    if "df" in st.session_state:
        pep = st.selectbox("Select peptide", st.session_state["df"]["Peptide"])
        st.pyplot(plot_helical_wheel(pep), use_container_width=False)
        st.pyplot(plot_hydropathy(pep), use_container_width=False)

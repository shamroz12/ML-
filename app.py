# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
import plotly.graph_objects as go

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
AA = list("ACDEFGHIKLMNPQRSTVWY")
hydro = {
"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,
"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,
"V":4.2,"W":-0.9,"Y":-1.3
}

# =========================
# Feature extraction
# =========================
def extract_features(seq):
    L = len(seq)
    aa_comp = [seq.count(a)/L for a in AA]
    dipep = [0]*400
    for i in range(L-1):
        dipep[AA.index(seq[i])*20 + AA.index(seq[i+1])] += 1
    dipep = [x/(L-1) if L>1 else 0 for x in dipep]
    return aa_comp + dipep + [L]

# =========================
# Screening proxies (CORE)
# =========================
def antigenicity_proxy(seq):
    # VaxiJen-like hydrophobicity + aromaticity proxy
    hydro_score = np.mean([hydro[a] for a in seq])
    aromatic = sum(a in "FWY" for a in seq)/len(seq)
    return hydro_score + aromatic

def conservancy_proxy(seq):
    # Low entropy proxy
    return 1 - (len(set(seq))/len(seq))

def mhc_binding_proxy(seq):
    # Anchor residue proxy (T-cell)
    anchors = ["L","I","V","F","Y","M"]
    return sum(a in anchors for a in seq)/len(seq)

def population_coverage_proxy(seq):
    # HLA-friendly residues
    return (seq.count("L")+seq.count("I")+seq.count("V"))/len(seq)

def structure_surface_proxy(seq):
    # Hydrophilicity → surface exposure proxy
    return -np.mean([hydro[a] for a in seq])

def toxicity_proxy(seq):
    return "High" if np.mean([hydro[a] for a in seq]) > 2.5 else "Low"

def allergenicity_proxy(seq):
    return "High" if seq.count("C")/len(seq) > 0.15 else "Low"

def epitope_type(seq):
    return "T-cell" if len(seq)>=13 else "B-cell"

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Prediction & Screening Platform")

fasta = st.text_area("Paste protein FASTA sequence:")
min_len = st.slider("Min length",8,15,9)
max_len = st.slider("Max length",9,25,15)
top_n = st.selectbox("Top N peptides",[20,50,100])

show_all = st.checkbox("Show all (PASS + FLAG)", value=False)

# =========================
# Run pipeline
# =========================
if st.button("Run Integrated Analysis"):

    seq = "".join([l for l in fasta.split() if not l.startswith(">")])

    peptides=[]
    starts=[]
    for L in range(min_len,max_len+1):
        for i in range(len(seq)-L+1):
            p = seq[i:i+L]
            if set(p).issubset(set(AA)):
                peptides.append(p)
                starts.append(i+1)

    feats = [extract_features(p) for p in peptides]
    X = pd.DataFrame(feats, columns=feature_columns)
    ml_scores = model.predict_proba(X)[:,1]

    rows=[]
    for pep,pos,ml in zip(peptides,starts,ml_scores):

        ag = antigenicity_proxy(pep)
        cons = conservancy_proxy(pep)
        mhc = mhc_binding_proxy(pep)
        pop = population_coverage_proxy(pep)
        surf = structure_surface_proxy(pep)
        tox = toxicity_proxy(pep)
        allerg = allergenicity_proxy(pep)
        etype = epitope_type(pep)

        reasons=[]
        if tox=="High": reasons.append("Toxicity")
        if allerg=="High": reasons.append("Allergenicity")
        if ag<0.5: reasons.append("Low antigenicity")
        if cons<0.3: reasons.append("Low conservancy")

        status = "PASS" if len(reasons)==0 else "FLAG"

        final_score = ml * ag * cons * mhc * pop * max(surf,0)

        rows.append([
            pep,pos,len(pep),etype,
            ml,ag,cons,mhc,pop,surf,
            tox,allerg,status,",".join(reasons),final_score
        ])

    df = pd.DataFrame(rows, columns=[
        "Peptide","Start","Length","Epitope_Type",
        "ML_Score","Antigenicity","Conservancy",
        "MHC_Binding","Population_Coverage","Surface_Exposure",
        "Toxicity","Allergenicity","Final_Status",
        "Rejection_Reasons","Final_Priority_Score"
    ])

    df = df.sort_values("Final_Priority_Score",ascending=False)

    if not show_all:
        df = df[df["Final_Status"]=="PASS"]

    df = df.head(top_n)

    st.subheader("📋 Integrated Decision Table")
    st.dataframe(df)

    # =========================
    # Visualization tracks
    # =========================
    density = np.zeros(len(seq))
    for _,r in df.iterrows():
        density[r.Start-1:r.Start+r.Length-1]+=1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(seq))), y=density, name="Epitope Density"))
    fig.update_layout(title="Epitope Density Track", xaxis_title="Protein Position")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Results",
        df.to_csv(index=False).encode("utf-8"),
        "integrated_epitope_results.csv"
    )

# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from sklearn.cluster import KMeans
from Bio import AlignIO

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

# =========================
# FASTA parser
# =========================
def read_fasta(text):
    lines = text.strip().splitlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq.upper()

# =========================
# UniProt domain fetch
# =========================
def fetch_uniprot_domains(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    r = requests.get(url)
    if r.status_code != 200:
        return []

    data = r.json()
    domains = []
    for feat in data.get("features", []):
        if feat["type"] in ["Domain", "Repeat", "Region"]:
            try:
                start = int(feat["location"]["start"]["value"])
                end = int(feat["location"]["end"]["value"])
                name = feat.get("description", feat["type"])
                domains.append((name, start, end))
            except:
                pass
    return domains

# =========================
# Conservation from MSA
# =========================
def conservation_from_msa(msa_file):
    aln = AlignIO.read(msa_file, "fasta")
    aln_len = aln.get_alignment_length()

    scores = []
    for i in range(aln_len):
        col = aln[:, i]
        freq = {}
        for a in col:
            if a != "-":
                freq[a] = freq.get(a, 0) + 1
        if len(freq) == 0:
            scores.append(0)
        else:
            m = max(freq.values())
            scores.append(m / sum(freq.values()))
    return np.array(scores)

# =========================
# UI
# =========================
st.title("🧬 Integrated Epitope Prioritization Platform")

fasta_input = st.text_area("Paste protein FASTA:")

uniprot_id = st.text_input("UniProt ID (optional, for domains):")

msa_file = st.file_uploader("Upload MSA FASTA (optional, for conservation):", type=["fasta","fa","faa"])

min_len = st.slider("Min peptide length", 8, 15, 9)
max_len = st.slider("Max peptide length", 9, 25, 15)

top_n = st.selectbox("Top N epitopes", [10,20,50,100])

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

    df = df.sort_values("Score", ascending=False).head(top_n)

    st.dataframe(df)

    # =========================
    # Clustering track
    # =========================
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[["Start","End","Score"]])

    # =========================
    # Domain track
    # =========================
    domains = fetch_uniprot_domains(uniprot_id) if uniprot_id else []

    # =========================
    # Conservation track
    # =========================
    if msa_file:
        cons = conservation_from_msa(msa_file)
    else:
        cons = np.zeros(len(seq))

    # =========================
    # Interactive Plotly Browser
    # =========================
    fig = go.Figure()

    # Protein backbone
    fig.add_trace(go.Scatter(x=[1,len(seq)], y=[0,0], mode="lines", line=dict(width=6), name="Protein"))

    # Epitopes
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[r.Start, r.End],
            y=[1,1],
            mode="lines",
            line=dict(width=10),
            name=f"Epi {r.Peptide}"
        ))

    # Conservation
    fig.add_trace(go.Scatter(
        x=list(range(1,len(cons)+1)),
        y=cons,
        name="Conservation",
        yaxis="y2"
    ))

    # Domains
    for name,s,e in domains:
        fig.add_trace(go.Scatter(
            x=[s,e],
            y=[-1,-1],
            mode="lines",
            line=dict(width=12),
            name=name
        ))

    fig.update_layout(
        title="Interactive Epitope Genome-Browser View",
        xaxis=dict(title="Protein position"),
        yaxis=dict(title="Epitopes", range=[-2,2]),
        yaxis2=dict(title="Conservation", overlaying="y", side="right"),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

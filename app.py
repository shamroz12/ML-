# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from itertools import product
import plotly.graph_objects as go
import requests
from sklearn.cluster import KMeans

# =========================
# Page config
# =========================
st.set_page_config(page_title="Advanced Epitope Prioritization Platform", layout="wide")

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
    total = len(seq)-1
    counts = {dp:0 for dp in dipeptides}
    for i in range(total):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp]+=1
    return [counts[dp]/total for dp in dipeptides] if total>0 else [0]*400

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
# UniProt domain fetch
# =========================
def fetch_uniprot_domains(uniprot):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.json"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    data = r.json()
    domains=[]
    for f in data.get("features",[]):
        if f["type"] in ["Domain","Region","Repeat"]:
            try:
                s=int(f["location"]["start"]["value"])
                e=int(f["location"]["end"]["value"])
                name=f.get("description",f["type"])
                domains.append((name,s,e))
            except:
                pass
    return domains

# =========================
# Proxies
# =========================
def classify_type(seq):
    if len(seq)>=13: return "T-cell"
    else: return "B-cell"

def conservation_proxy(seq):
    return 1.0 - (len(set(seq))/len(seq))

def population_coverage_proxy(seq):
    return min(1.0, (seq.count("L")+seq.count("I")+seq.count("V"))/len(seq))

# =========================
# UI
# =========================
st.title("🧬 Advanced Epitope Prioritization & Visualization Platform")

fasta = st.text_area("Paste protein FASTA sequence:")
uniprot_id = st.text_input("UniProt ID (optional, for domain overlay):")

min_len = st.slider("Min length",8,15,9)
max_len = st.slider("Max length",9,25,15)
top_n = st.selectbox("Top N epitopes",[10,20,50,100])

# =========================
# Predict
# =========================
if st.button("🔍 Run Full Epitope Analysis"):

    seq = read_fasta(fasta)

    peptides=[]
    starts=[]
    for L in range(min_len,max_len+1):
        for i in range(len(seq)-L+1):
            p = seq[i:i+L]
            if set(p).issubset(set(amino_acids)):
                peptides.append(p)
                starts.append(i+1)

    feats=[extract_features(p) for p in peptides]
    X=pd.DataFrame(feats,columns=feature_columns)
    probs=model.predict_proba(X)[:,1]

    df=pd.DataFrame({
        "Peptide":peptides,
        "Start":starts,
        "Length":[len(p) for p in peptides],
        "Score":probs
    })
    df["End"]=df["Start"]+df["Length"]-1

    df["Type"]=df["Peptide"].apply(classify_type)
    df["Conservation"]=df["Peptide"].apply(conservation_proxy)
    df["Population_Coverage"]=df["Peptide"].apply(population_coverage_proxy)

    df=df.sort_values("Score",ascending=False).head(top_n)

    # Clustering
    km=KMeans(n_clusters=3,n_init=10,random_state=42)
    df["Cluster"]=km.fit_predict(df[["Start","End","Score"]])

    st.subheader("📋 Final Prioritized Epitopes")
    st.dataframe(df)

    # =========================
    # Tracks
    # =========================
    domains = fetch_uniprot_domains(uniprot_id) if uniprot_id else []

    # Density track
    density = np.zeros(len(seq))
    for _,r in df.iterrows():
        density[r.Start-1:r.End]+=1

    # =========================
    # Plot
    # =========================
    fig=go.Figure()

    # Protein backbone
    fig.add_trace(go.Scatter(x=[1,len(seq)],y=[0,0],mode="lines",line=dict(width=6),name="Protein"))

    # Epitopes
    colors=["red","blue","green","orange","purple"]
    for _,r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[r.Start,r.End],
            y=[1,1],
            mode="lines",
            line=dict(width=10,color=colors[int(r.Cluster)%5]),
            name=r.Peptide,
            hovertext=f"""
Score:{r.Score:.3f}
Type:{r.Type}
Conservation:{r.Conservation:.2f}
Population:{r.Population_Coverage:.2f}
"""
        ))

    # Domains
    for name,s,e in domains:
        fig.add_trace(go.Scatter(x=[s,e],y=[-1,-1],mode="lines",line=dict(width=12),name=f"Domain: {name}"))

    # Density
    fig.add_trace(go.Scatter(
        x=list(range(1,len(seq)+1)),
        y=density,
        name="Epitope Density",
        yaxis="y2",
        line=dict(color="black")
    ))

    fig.update_layout(
        title="🧬 Interactive Epitope Landscape Browser",
        xaxis=dict(title="Protein Position"),
        yaxis=dict(visible=False,range=[-2,2]),
        yaxis2=dict(overlaying="y",side="right",title="Density"),
        height=700
    )

    st.plotly_chart(fig,use_container_width=True)

    # =========================
    # Download
    # =========================
    st.download_button(
        "⬇️ Download Full Results",
        df.to_csv(index=False).encode("utf-8"),
        "final_epitope_analysis.csv",
        "text/csv"
    )

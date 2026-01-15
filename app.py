# =========================
# Unified Epitope Intelligence & Vaccine Design Platform (FULL INDUSTRIAL VERSION)
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
# Page config
# =========================
st.set_page_config(page_title="Unified Epitope & Vaccine Design Platform", layout="wide")

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

charge = {
"A":0,"C":0,"D":-1,"E":-1,"F":0,"G":0,"H":0.1,"I":0,"K":1,"L":0,
"M":0,"N":0,"P":0,"Q":0,"R":1,"S":0,"T":0,"V":0,"W":0,"Y":0
}

aromatic_set = set("FWY")
aliphatic = set("AVLIM")

# =========================
# Feature extraction (ML)
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
# Proxies
# =========================
def antigenicity_proxy(seq):
    return sum(hydro[a] for a in seq)/len(seq)

def cell_type_proxy(seq):
    L=len(seq)
    hydv=sum(hydro[a] for a in seq)/L
    if 8<=L<=11 and hydv>1: return "T-cell"
    if hydv<0: return "B-cell"
    return "Both"

# =========================
# FASTA
# =========================
def read_fasta_multi(text):
    seqs=[]; cur=""
    for l in text.strip().splitlines():
        if l.startswith(">"):
            if cur: seqs.append(cur); cur=""
        else: cur+=l.strip()
    if cur: seqs.append(cur)
    return [s.upper() for s in seqs]

# =========================
# Conservancy
# =========================
def conservancy_percent(pep, seqs):
    return 100*sum(pep in s for s in seqs)/len(seqs)

# =========================
# =========================
# ADVANCED PEPTIDE CHEMISTRY ENGINE
# =========================
# =========================

def hydrophobic_moment(seq):
    L=len(seq)
    angles=[i*100*pi/180 for i in range(L)]
    mx=sum(hydro[seq[i]]*cos(angles[i]) for i in range(L))
    my=sum(hydro[seq[i]]*sin(angles[i]) for i in range(L))
    return (mx*mx+my*my)**0.5 / L

def gravy(seq):
    return sum(hydro[a] for a in seq)/len(seq)

def aliphatic_index(seq):
    return 100 * sum(a in aliphatic for a in seq) / len(seq)

def boman_index(seq):
    return -sum(hydro[a] for a in seq)/len(seq)

def instability_index(seq):
    return 10 * sum(seq[i]==seq[i+1] for i in range(len(seq)-1))

def membrane_binding_prob(seq):
    x = 0.8*hydrophobic_moment(seq) + 0.5*gravy(seq)
    return 1/(1+exp(-3*x))

def cpp_score(seq):
    return (sum(a in "KR" for a in seq)/len(seq)) * 10

def aggregation_score(seq):
    return max(0, gravy(seq)) + sum(a in "IVLFWY" for a in seq)/len(seq)

def toxicity_score(seq):
    return (sum(a in "KR" for a in seq)/len(seq)) * gravy(seq)**2

def protease_stability(seq):
    cuts = sum(a in "KR" for a in seq[1:-1])
    return 1 / (1 + cuts)

def solubility_score(seq):
    return 1 / (1 + exp(gravy(seq)))

def immunogenicity_solubility_tradeoff(final_ml, seq):
    return 0.6*final_ml + 0.4*solubility_score(seq)

# =========================
# Helical wheel
# =========================
def plot_helical_wheel(seq):
    L=len(seq)
    angles=[i*100*pi/180 for i in range(L)]
    fig, ax = plt.subplots(figsize=(6,6))
    xs=[]; ys=[]
    for i,a in enumerate(angles):
        x=cos(a); y=sin(a)
        xs.append(x); ys.append(y)
        aa=seq[i]
        c="red" if hydro[aa]>1 else "blue" if hydro[aa]<-1 else "green"
        ax.scatter(x,y,s=1000,c=c,alpha=0.7)
        ax.text(x,y,aa,ha="center",va="center",color="white",fontweight="bold")
    mx=sum(hydro[seq[i]]*xs[i] for i in range(L))
    my=sum(hydro[seq[i]]*ys[i] for i in range(L))
    ax.arrow(0,0,mx/5,my/5,head_width=0.1,color="black",linewidth=3)
    circle=plt.Circle((0,0),1,fill=False,linestyle="dashed")
    ax.add_artist(circle)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Helical Wheel Projection")
    return fig

def plot_hydropathy(seq):
    vals=[hydro[a] for a in seq]
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(range(1,len(seq)+1),vals,marker="o")
    ax.axhline(0,linestyle="--")
    ax.set_title("Hydropathy Plot")
    ax.set_xlabel("Position")
    ax.set_ylabel("Hydrophobicity")
    return fig

# =========================
# =========================
# UI
# =========================
# =========================

st.title("🧬 Unified Epitope Intelligence & Vaccine Design Platform")

tabs = st.tabs(["Pipeline","SHAP","Vaccine","Landscape","3D","Chemistry","Export","Report"])

# =========================
# PIPELINE
# =========================
with tabs[0]:
    fasta_input = st.text_area("Paste FASTA:")
    min_len = st.slider("Min length",8,15,9)
    max_len = st.slider("Max length",9,25,15)
    top_n = st.selectbox("Top N",[10,20,50,100])

    if st.button("Run Pipeline"):
        seqs = read_fasta_multi(fasta_input)
        main = seqs[0]

        peptides=[]; positions=[]
        for L in range(min_len,max_len+1):
            for i in range(len(main)-L+1):
                pep=main[i:i+L]
                peptides.append(pep); positions.append(i+1)

        X = pd.DataFrame([extract_features(p) for p in peptides], columns=feature_columns)
        probs = model.predict_proba(X)[:,1]

        rows=[]
        for pep,pos,ml in zip(peptides,positions,probs):
            cons=conservancy_percent(pep,seqs)
            antig=antigenicity_proxy(pep)
            final=0.6*ml + 0.3*(cons/100) + 0.1*(antig/5)

            rows.append([pep,pos,len(pep),ml,cons,antig,final,cell_type_proxy(pep)])

        df=pd.DataFrame(rows,columns=["Peptide","Start","Length","ML","Conservancy_%","Antigenicity","FinalScore","Cell_Type"])
        df=df.sort_values("FinalScore",ascending=False).head(top_n)

        st.session_state["df"]=df
        st.session_state["X"]=X
        st.dataframe(df)

# =========================
# SHAP
# =========================
with tabs[1]:
    if "df" in st.session_state:
        X=st.session_state["X"]
        explainer=shap.TreeExplainer(model)
        shap_vals=explainer.shap_values(X.iloc[:20])
        fig,_=plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_vals, X.iloc[:20], show=False)
        st.pyplot(fig)

# =========================
# VACCINE
# =========================
with tabs[2]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        construct="GPGPG".join(df["Peptide"].tolist())
        st.code(construct)

# =========================
# LANDSCAPE
# =========================
with tabs[3]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        fig,ax=plt.subplots(figsize=(12,4))
        ax.scatter(df["Start"],df["FinalScore"],c=df["FinalScore"],cmap="viridis",s=80)
        ax.set_title("Immunogenic Landscape")
        st.pyplot(fig)

# =========================
# CHEMISTRY
# =========================
with tabs[5]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        pep = st.selectbox("Select peptide", df["Peptide"])
        ml_score = df[df["Peptide"]==pep]["FinalScore"].values[0]

        st.subheader("🧪 Full Developability & Chemistry Analysis")
        st.code(pep)

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("GRAVY", f"{gravy(pep):.2f}")
        col2.metric("Hydrophobic Moment", f"{hydrophobic_moment(pep):.3f}")
        col3.metric("Aliphatic Index", f"{aliphatic_index(pep):.1f}")
        col4.metric("Boman Index", f"{boman_index(pep):.2f}")

        col1.metric("Membrane Binding Prob", f"{membrane_binding_prob(pep):.2f}")
        col2.metric("CPP Score", f"{cpp_score(pep):.2f}")
        col3.metric("Aggregation Risk", f"{aggregation_score(pep):.2f}")
        col4.metric("Toxicity Risk", f"{toxicity_score(pep):.2f}")

        col1.metric("Protease Stability", f"{protease_stability(pep):.2f}")
        col2.metric("Solubility", f"{solubility_score(pep):.2f}")
        col3.metric("Multi-objective Score", f"{immunogenicity_solubility_tradeoff(ml_score, pep):.3f}")

        st.subheader("🌀 Helical Wheel")
        st.pyplot(plot_helical_wheel(pep))

        st.subheader("📈 Hydropathy Plot")
        st.pyplot(plot_hydropathy(pep))

# =========================
# EXPORT
# =========================
with tabs[6]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        csv=df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV",csv,"epitopes.csv")

# =========================
# REPORT
# =========================
with tabs[7]:
    if "df" in st.session_state:
        if st.button("Generate PDF"):
            df=st.session_state["df"]
            fig,ax=plt.subplots()
            ax.scatter(df["Start"],df["FinalScore"])
            with PdfPages("Report.pdf") as pdf:
                pdf.savefig(fig)
            with open("Report.pdf","rb") as f:
                st.download_button("⬇️ Download PDF", f, "Epitope_Report.pdf")

# =========================
# Unified Epitope Intelligence & Vaccine Design Platform
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

SIGNAL_PEPTIDES = {
    "None": "",
    "tPA": "MDAMKRGLCCVLLLCGAVFVS",
    "IL2": "MYRMQLLSCIALSLALVTNS"
}

ADJUVANTS = {
    "None": "",
    "β-defensin": "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRK"
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
# Biology proxies
# =========================
def toxicity_proxy(seq):
    return "High" if sum(hydro[a] for a in seq)/len(seq) > 2.5 else "Low"

def allergenicity_proxy(seq):
    aromatic_frac = sum(a in "FWY" for a in seq)/len(seq)
    cyst_frac = seq.count("C")/len(seq)
    return "High" if aromatic_frac>0.3 or cyst_frac>0.15 else "Low"

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
# Conservancy & robustness
# =========================
def conservancy_percent(pep, seqs):
    return 100*sum(pep in s for s in seqs)/len(seqs)

def robustness_score(pep, seqs):
    L=len(pep); hits=0
    for i in range(L):
        for a in amino_acids:
            if a!=pep[i]:
                mut=pep[:i]+a+pep[i+1:]
                if any(mut in s for s in seqs): hits+=1
    return hits/(L*19)

# =========================
# Population proxy
# =========================
def population_coverage_proxy(seq):
    L=len(seq)
    if 9<=L<=11: return "Broad",0.9
    if L>=8: return "Medium",0.6
    return "Narrow",0.3

# =========================
# 3D Viewer
# =========================
def show_structure_3d_advanced(pdb_text, df):
    view = py3Dmol.view(width=1000, height=700)
    view.addModel(pdb_text, "pdb")

    style = st.selectbox("Style", ["cartoon","surface","stick","sphere"])
    color_mode = st.selectbox("Color by", ["score","cell","conservancy","uniform"])
    focus = st.selectbox("Focus epitope", ["ALL"] + df["Peptide"].tolist())

    view.setStyle({"cartoon":{"color":"lightgray"}})

    scores=df["FinalScore"].values
    cons=df["Conservancy_%"].values
    smin,smax=scores.min(),scores.max()
    cmin,cmax=cons.min(),cons.max()

    def score_color(x):
        t=(x-smin)/(smax-smin+1e-6)
        return f"rgb({int(255*t)},0,{int(255*(1-t))})"

    def cons_color(x):
        t=(x-cmin)/(cmax-cmin+1e-6)
        return f"rgb(0,{int(255*t)},0)"

    cell_colors={"T-cell":"red","B-cell":"blue","Both":"purple"}

    for _,r in df.iterrows():
        pep=r["Peptide"]
        if focus!="ALL" and pep!=focus: continue

        s=int(r["Start"])
        e=int(r["Start"]+r["Length"])

        if color_mode=="score": col=score_color(r["FinalScore"])
        elif color_mode=="conservancy": col=cons_color(r["Conservancy_%"])
        elif color_mode=="cell": col=cell_colors[r["Cell_Type"]]
        else: col="orange"

        view.setStyle({"resi":list(range(s,e+1))},{"cartoon":{"color":col}})

    view.zoomTo()
    components.html(view._make_html(), height=750, scrolling=False)

# =========================
# Vaccine construct plot
# =========================
def draw_vaccine_construct_figure(df, linker="GPGPG", signal="", adjuvant=""):
    fig, ax = plt.subplots(figsize=(18,4), dpi=300)
    colors={"T-cell":"#d62728","B-cell":"#1f77b4","Both":"#9467bd"}

    x=0; y=0.5; h=0.3

    if signal:
        ax.add_patch(plt.Rectangle((x,y),len(signal),h,color="#2ca02c")); x+=len(signal)
    if adjuvant:
        ax.add_patch(plt.Rectangle((x,y),len(adjuvant),h,color="#ff7f0e")); x+=len(adjuvant)

    for i,r in df.iterrows():
        L=len(r["Peptide"])
        ax.add_patch(plt.Rectangle((x,y),L,h,color=colors[r["Cell_Type"]]))
        ax.text(x+L/2,y+h/2,r["Peptide"],ha="center",va="center",rotation=90,fontsize=7,color="white")
        x+=L
        if i!=df.index[-1]:
            ax.add_patch(plt.Rectangle((x,y),len(linker),h,color="gray"))
            x+=len(linker)

    ax.set_xlim(0,x); ax.set_ylim(0,1.2); ax.set_yticks([])
    ax.set_title("Multi-Epitope Vaccine Construct Architecture")
    return fig

# =========================
# UI
# =========================
st.title("🧬 Unified Epitope Intelligence & Vaccine Design Platform")

tabs = st.tabs(["Pipeline","SHAP","Vaccine","Landscape","3D","Export","Report"])

# =========================
# TAB 1 — PIPELINE
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
                if set(pep).issubset(set(amino_acids)):
                    peptides.append(pep); positions.append(i+1)

        X = pd.DataFrame([extract_features(p) for p in peptides], columns=feature_columns)

        preds=[]
        for _ in range(5):
            idx=np.random.choice(len(X),len(X),replace=True)
            preds.append(model.predict_proba(X.iloc[idx])[:,1])
        preds=np.array(preds)
        meanp=preds.mean(axis=0); stdp=preds.std(axis=0)

        rows=[]
        for i,(pep,pos,ml) in enumerate(zip(peptides,positions,meanp)):
            cons=conservancy_percent(pep,seqs)
            rob=robustness_score(pep,seqs)
            popc,pops=population_coverage_proxy(pep)
            antig=antigenicity_proxy(pep)

            final=0.35*ml+0.25*(cons/100)+0.2*rob+0.1*pops+0.1*(antig/5)

            rows.append([pep,pos,len(pep),ml,stdp[i],cons,rob,antig,popc,pops,final,cell_type_proxy(pep)])

        df=pd.DataFrame(rows,columns=["Peptide","Start","Length","ML","ML_std","Conservancy_%","Robustness","Antigenicity","PopClass","PopScore","FinalScore","Cell_Type"])
        df=df.sort_values("FinalScore",ascending=False).head(top_n)

        st.session_state["df"]=df
        st.session_state["X"]=X
        st.dataframe(df)

# =========================
# TAB 2 — SHAP
# =========================
with tabs[1]:
    if "df" in st.session_state:
        X=st.session_state["X"]
        explainer=shap.TreeExplainer(model)
        shap_vals=explainer.shap_values(X.iloc[:5])
        fig,_=plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_vals, X.iloc[:5], show=False)
        st.pyplot(fig)

# =========================
# TAB 3 — VACCINE
# =========================
with tabs[2]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        sig=st.selectbox("Signal",list(SIGNAL_PEPTIDES.keys()))
        adj=st.selectbox("Adjuvant",list(ADJUVANTS.keys()))
        linker="GPGPG"

        construct = SIGNAL_PEPTIDES[sig] + ADJUVANTS[adj] + linker.join(df["Peptide"].tolist())
        st.code(construct)

        fig=draw_vaccine_construct_figure(df,linker,SIGNAL_PEPTIDES[sig],ADJUVANTS[adj])
        st.pyplot(fig)

# =========================
# TAB 4 — LANDSCAPE
# =========================
with tabs[3]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        fig,ax=plt.subplots(figsize=(12,4))
        ax.scatter(df["Start"],df["FinalScore"],c=df["FinalScore"],cmap="viridis",s=80)
        ax.set_title("Immunogenic Landscape")
        st.pyplot(fig)

# =========================
# TAB 5 — 3D
# =========================
with tabs[4]:
    pdb_file = st.file_uploader("Upload PDB", type=["pdb"])
    if "df" in st.session_state and pdb_file:
        show_structure_3d_advanced(pdb_file.read().decode("utf-8"), st.session_state["df"])

# =========================
# TAB 6 — EXPORT
# =========================
with tabs[5]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        st.download_button("Download CSV", df.to_csv(index=False), "epitopes.csv")

# =========================
# TAB 7 — REPORT
# =========================
with tabs[6]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        if st.button("Generate PDF"):
            fig, ax = plt.subplots()
            ax.scatter(df["Start"], df["FinalScore"])
            ax.set_title("Immunogenic Landscape")

            with PdfPages("Report.pdf") as pdf:
                pdf.savefig(fig)

            st.success("Report generated!")

            # 👇 ADD THIS
            with open("Report.pdf", "rb") as f:
                st.download_button(
                    label="📥 Download Report.pdf",
                    data=f,
                    file_name="Epitope_Report.pdf",
                    mime="application/pdf"
                )


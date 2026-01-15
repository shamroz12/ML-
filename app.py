# =========================
# Unified Epitope Intelligence & Vaccine Design Platform (FINAL CLEAN VERSION)
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
# GLOBAL PLOT QUALITY SETTINGS
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

charge = {
"A":0,"C":0,"D":-1,"E":-1,"F":0,"G":0,"H":0.1,"I":0,"K":1,"L":0,
"M":0,"N":0,"P":0,"Q":0,"R":1,"S":0,"T":0,"V":0,"W":0,"Y":0
}

aromatic_set = set("FWY")
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
# 3D Viewer
# =========================
def show_structure_3d(pdb_text, df):
    view = py3Dmol.view(width=900, height=600)
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
        if focus!="ALL" and pep!=focus:
            continue

        s=int(r["Start"])
        e=int(r["Start"]+r["Length"])

        if color_mode=="score": col=score_color(r["FinalScore"])
        elif color_mode=="conservancy": col=cons_color(r["Conservancy_%"])
        elif color_mode=="cell": col=cell_colors[r["Cell_Type"]]
        else: col="orange"

        view.setStyle({"resi":list(range(s,e+1))},{"cartoon":{"color":col}})

    view.zoomTo()
    components.html(view._make_html(), height=650, scrolling=False)

# =========================
# Chemistry Engine
# =========================
def hydrophobic_moment(seq):
    L=len(seq)
    angles=[i*100*pi/180 for i in range(L)]
    mx=sum(hydro[seq[i]]*cos(angles[i]) for i in range(L))
    my=sum(hydro[seq[i]]*sin(angles[i]) for i in range(L))
    return (mx*mx+my*my)**0.5 / L

def gravy(seq): return sum(hydro[a] for a in seq)/len(seq)
def aliphatic_index(seq): return 100 * sum(a in aliphatic for a in seq) / len(seq)
def boman_index(seq): return -sum(hydro[a] for a in seq)/len(seq)
def instability_index(seq): return 10 * sum(seq[i]==seq[i+1] for i in range(len(seq)-1))
def membrane_binding_prob(seq):
    x = 0.8*hydrophobic_moment(seq) + 0.5*gravy(seq)
    return 1/(1+exp(-3*x))
def cpp_score(seq): return (sum(a in "KR" for a in seq)/len(seq)) * 10
def aggregation_score(seq): return max(0, gravy(seq)) + sum(a in "IVLFWY" for a in seq)/len(seq)
def toxicity_score(seq): return (sum(a in "KR" for a in seq)/len(seq)) * gravy(seq)**2
def protease_stability(seq):
    cuts = sum(a in "KR" for a in seq[1:-1])
    return 1 / (1 + cuts)
def solubility_score(seq): return 1 / (1 + exp(gravy(seq)))
def immunogenicity_solubility_tradeoff(final_ml, seq): return 0.6*final_ml + 0.4*solubility_score(seq)

# =========================
# Plots (FIXED SIZE + DPI)
# =========================
def plot_helical_wheel(seq):
    L=len(seq)
    angles=[i*100*pi/180 for i in range(L)]
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)

    xs=[]; ys=[]
    for i,a in enumerate(angles):
        x=cos(a); y=sin(a)
        xs.append(x); ys.append(y)
        aa=seq[i]
        c="red" if hydro[aa]>1 else "blue" if hydro[aa]<-1 else "green"
        ax.scatter(x,y,s=600,c=c,alpha=0.85,edgecolors="black",linewidths=0.5)
        ax.text(x,y,aa,ha="center",va="center",color="white",fontweight="bold",fontsize=11)

    mx=sum(hydro[seq[i]]*xs[i] for i in range(L))
    my=sum(hydro[seq[i]]*ys[i] for i in range(L))
    ax.arrow(0,0,mx/5,my/5, head_width=0.08, linewidth=2, color="black")

    circle=plt.Circle((0,0),1,fill=False,linestyle="dashed",linewidth=1)
    ax.add_artist(circle)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Helical Wheel Projection", fontsize=12)
    return fig

def plot_hydropathy(seq):
    vals=[hydro[a] for a in seq]
    fig,ax=plt.subplots(figsize=(6,2.5), dpi=200)
    ax.plot(range(1,len(seq)+1),vals,marker="o")
    ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_title("Hydropathy Plot", fontsize=11)
    ax.set_xlabel("Position")
    ax.set_ylabel("Hydrophobicity")
    return fig
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
# UI
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
        fig,_=plt.subplots(figsize=(7,4), dpi=200)
        shap.summary_plot(shap_vals, X.iloc[:20], show=False)
        st.pyplot(fig, use_container_width=False)

# =========================
# VACCINE DESIGNER (ADVANCED)
# =========================
with tabs[2]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        st.title("🧬 Intelligent Multi-Epitope Vaccine Designer")

        colA, colB, colC = st.columns(3)

        with colA:
            sig = st.selectbox("Signal peptide", list(SIGNAL_PEPTIDES.keys()))
            adj = st.selectbox("Adjuvant", list(ADJUVANTS.keys()))
            linker = st.selectbox("Linker", ["GPGPG", "AAY", "EAAAK"])

        with colB:
            n_epi = st.slider("Number of epitopes", 3, min(20, len(df)), min(8, len(df)))
            cell_filter = st.selectbox("Epitope type", ["All", "T-cell", "B-cell", "Both"])

        with colC:
            ordering = st.selectbox("Order epitopes by", ["FinalScore", "Conservancy_%", "Start"])
            add_padre = st.checkbox("Add PADRE helper epitope", value=True)
            filter_bad = st.checkbox("Remove toxic / aggregating peptides", value=True)

        work = df.copy()
        if cell_filter != "All":
            work = work[work["Cell_Type"] == cell_filter]

        if filter_bad:
            work = work[work["Peptide"].apply(lambda p: aggregation_score(p) < 1.5 and toxicity_score(p) < 1)]

        work = work.sort_values(ordering, ascending=False)
        selected = work.head(n_epi)["Peptide"].tolist()

        blocks = []
        if sig != "None":
            blocks.append(("Signal", SIGNAL_PEPTIDES[sig]))
        if adj != "None":
            blocks.append(("Adjuvant", ADJUVANTS[adj]))
        for p in selected:
            blocks.append(("Epitope", p))
        if add_padre:
            blocks.append(("PADRE", PADRE))

        seq_parts = [b[1] for b in blocks]
        construct = linker.join(seq_parts)

        st.subheader("🧱 Vaccine Architecture")
        cols = st.columns(len(blocks))
        for (label, seq), c in zip(blocks, cols):
            color = "#4CAF50" if label=="Epitope" else "#FF9800" if label=="Adjuvant" else "#03A9F4" if label=="Signal" else "#9C27B0"
            c.markdown(
                f"""
                <div style="padding:10px;border-radius:10px;background:{color};color:white;text-align:center">
                <b>{label}</b><br>{len(seq)} aa
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("🧬 Final Vaccine Construct")
        st.code(construct)
        
        st.subheader("📊 Construct Quality Metrics")
        qm = construct_quality_metrics(selected if len(selected) > 0 else None)
        if qm:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Avg GRAVY", f"{qm['Avg GRAVY']:.2f}")
            c2.metric("Avg Solubility", f"{qm['Avg Solubility']:.2f}")
            c3.metric("Avg Aggregation", f"{qm['Avg Aggregation']:.2f}")
            c4.metric("Avg Membrane Bind", f"{qm['Avg Membrane Binding']:.2f}")
            c5.metric("Developability", f"{qm['Developability Score']:.2f}")

        st.subheader("⬇️ Export")
        fasta = f">Multi_epitope_vaccine\n{construct}"
        st.download_button("Download FASTA", fasta, "vaccine.fasta")
        st.download_button("Download Sequence TXT", construct, "vaccine.txt")

# =========================
# LANDSCAPE
# =========================
with tabs[3]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        fig,ax=plt.subplots(figsize=(7,3), dpi=200)
        ax.scatter(df["Start"],df["FinalScore"],c=df["FinalScore"],cmap="viridis",s=60)
        ax.set_title("Immunogenic Landscape", fontsize=11)
        st.pyplot(fig, use_container_width=False)

# =========================
# 3D
# =========================
with tabs[4]:
    pdb_file = st.file_uploader("Upload PDB", type=["pdb"])
    if "df" in st.session_state and pdb_file:
        show_structure_3d(pdb_file.read().decode("utf-8"), st.session_state["df"])

# =========================
# CHEMISTRY
# =========================
with tabs[5]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        pep = st.selectbox("Select peptide", df["Peptide"])
        ml_score = df[df["Peptide"]==pep]["FinalScore"].values[0]

        st.subheader("🧪 Peptide Chemistry")
        st.code(pep)

        st.pyplot(plot_helical_wheel(pep), use_container_width=False)
        st.pyplot(plot_hydropathy(pep), use_container_width=False)

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
            fig,ax=plt.subplots(figsize=(6,4), dpi=200)
            ax.scatter(df["Start"],df["FinalScore"])
            with PdfPages("Report.pdf") as pdf:
                pdf.savefig(fig)
            with open("Report.pdf","rb") as f:
                st.download_button("⬇️ Download PDF", f, "Epitope_Report.pdf")

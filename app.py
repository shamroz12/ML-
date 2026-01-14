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
    hydv = sum(hydro.get(a,0) for a in seq)/L
    aromatic = sum(a in "FWY" for a in seq)/L
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
# 3D viewer
# =========================
def show_structure_3d(pdb_text, df, mode="ALL", style="cartoon", color_mode="score"):
    view = py3Dmol.view(width=900, height=600)
    view.addModel(pdb_text, "pdb")

    if style=="cartoon": view.setStyle({"cartoon":{"color":"lightgray"}})
    if style=="surface": view.setStyle({"surface":{"opacity":0.9,"color":"lightgray"}})
    if style=="stick": view.setStyle({"stick":{}})

    scores=df["FinalScore"].values
    mn, mx = scores.min(), scores.max()

    def score_color(s):
        x=(s-mn)/(mx-mn+1e-6)
        r=int(255*x); b=int(255*(1-x))
        return f"rgb({r},0,{b})"

    if mode!="ALL":
        df = df[df["Peptide"]==mode]

    for _,r in df.iterrows():
        s=int(r["Start"]); e=int(r["Start"]+r["Length"])
        if color_mode=="score": col=score_color(r["FinalScore"])
        elif color_mode=="cell":
            col={"T-cell":"red","B-cell":"blue","Both":"purple"}[r["Cell_Type"]]
        else: col="orange"

        view.setStyle({"resi":list(range(s,e+1))},{style:{"color":col}})

    view.zoomTo()
    components.html(view._make_html(), height=650, scrolling=False)

# =========================
# Vaccine construct drawing
# =========================
def draw_vaccine_construct_figure(df, linker="GPGPG", save_path=None):

    fig, ax = plt.subplots(figsize=(16, 3), dpi=300)

    colors = {"T-cell":"#d62728","B-cell":"#1f77b4","Both":"#9467bd"}

    x = 0; y = 0.5; height = 0.3; total_len = 0

    for i, r in df.iterrows():
        pep = r["Peptide"]; L = len(pep); ctype = r["Cell_Type"]

        ax.add_patch(plt.Rectangle((x, y), L, height, facecolor=colors.get(ctype,"black"), edgecolor="black"))
        ax.text(x+L/2, y+height/2, pep, ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")

        x += L; total_len += L

        if i != df.index[-1]:
            Lk = len(linker)
            ax.add_patch(plt.Rectangle((x, y), Lk, height, facecolor="#7f7f7f", edgecolor="black", hatch="//"))
            ax.text(x+Lk/2, y+height/2, linker, ha="center", va="center", fontsize=7, rotation=90, color="white")
            x += Lk; total_len += Lk

    ax.set_xlim(0, x); ax.set_ylim(0, 1.2); ax.set_yticks([])
    ax.set_xlabel("Amino acid position"); ax.set_title("Multi-epitope Vaccine Construct Architecture", fontsize=16, fontweight="bold")
    ax.plot([0, total_len], [0.25, 0.25], color="black", lw=2)
    ax.text(total_len/2, 0.18, f"Total length = {total_len} aa", ha="center")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
# =========================
# Construct optimization
# =========================

def junction_penalty(seq):
    # penalize hydrophobic-hydrophobic junctions
    penalty = 0
    for i in range(len(seq)-1):
        if hydro.get(seq[i],0) > 1 and hydro.get(seq[i+1],0) > 1:
            penalty += 1
    return penalty / len(seq)

def hydrophobic_cluster_penalty(seq):
    # penalize long hydrophobic stretches
    penalty = 0
    run = 0
    for a in seq:
        if hydro.get(a,0) > 1:
            run += 1
            if run >= 4:
                penalty += run
        else:
            run = 0
    return penalty / len(seq)

def repeat_penalty(seq):
    # penalize AAA, GGG etc
    penalty = 0
    for i in range(len(seq)-2):
        if seq[i] == seq[i+1] == seq[i+2]:
            penalty += 1
    return penalty / len(seq)

def construct_fitness(epitopes, df, linker="GPGPG"):
    seq = linker.join(epitopes)

    mean_score = df.set_index("Peptide").loc[epitopes]["FinalScore"].mean()

    jpen = junction_penalty(seq)
    hpen = hydrophobic_cluster_penalty(seq)
    rpen = repeat_penalty(seq)

    fitness = (
        0.4 * mean_score
        - 0.3 * jpen
        - 0.2 * hpen
        - 0.1 * rpen
    )

    return fitness, seq, jpen, hpen, rpen

def optimize_construct(df, linker="GPGPG", n_iter=500):

    epitopes = df["Peptide"].tolist()
    best_order = epitopes.copy()
    best_fitness, best_seq, best_j, best_h, best_r = construct_fitness(best_order, df, linker)

    for _ in range(n_iter):
        trial = best_order.copy()
        np.random.shuffle(trial)

        fit, seq, j, h, r = construct_fitness(trial, df, linker)

        if fit > best_fitness:
            best_fitness = fit
            best_order = trial
            best_seq = seq
            best_j = j
            best_h = h
            best_r = r

    return best_order, best_seq, best_fitness, best_j, best_h, best_r

# =========================
# UI
# =========================
st.title("🧬 Unified Epitope Intelligence & Vaccine Design Platform")

tabs = st.tabs(["🔬 Pipeline","🧠 SHAP","🧩 Vaccine Construct","🗺️ Landscape","🧬 3D Structure","🧱 Export","📄 Report"])

# ---------------- TAB 1 ----------------
with tabs[0]:
    fasta_input = st.text_area("Paste FASTA (variants allowed):")
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
        meanp = model.predict_proba(X)[:,1]

        rows=[]
        for pep,pos,ml in zip(peptides,positions,meanp):
            tox=toxicity_proxy(pep); allerg=allergenicity_proxy(pep)
            antig=antigenicity_proxy(pep); cell=cell_type_proxy(pep)
            cons=conservancy_percent(pep,seqs); rob=robustness_score(pep,seqs)
            popc,pops=population_coverage_proxy(pep)
            final=0.4*ml+0.25*(cons/100)+0.2*rob+0.1*pops+0.05*(antig/5)
            rows.append([pep,pos,len(pep),ml,cons,rob,antig,popc,pops,final,tox,allerg,cell])

        df=pd.DataFrame(rows,columns=["Peptide","Start","Length","ML","Conservancy_%","Robustness","Antigenicity","PopClass","PopScore","FinalScore","Toxicity","Allergenicity","Cell_Type"])
        df=df.sort_values("FinalScore",ascending=False).head(top_n)
        st.session_state["df"]=df; st.session_state["X"]=X
        st.dataframe(df)

# ---------------- TAB 2 ----------------
with tabs[1]:
    if "df" in st.session_state:
        X=st.session_state["X"]
        explainer=shap.TreeExplainer(model)
        shap_vals=explainer.shap_values(X.iloc[:5])
        fig,_=plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_vals, X.iloc[:5], show=False)
        st.pyplot(fig)

# ---------------- TAB 3 ----------------
with tabs[2]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("🧩 Vaccine Construct Designer & Optimizer")

        linker = "GPGPG"

        st.markdown("### 🔹 Baseline Construct (Ranked Order)")
        baseline_order = df["Peptide"].tolist()
        baseline_seq = linker.join(baseline_order)

        st.code(baseline_seq)
        st.write("Length:", len(baseline_seq), "aa")

        if st.button("🚀 Optimize Construct Order"):

            with st.spinner("Optimizing epitope order..."):
                best_order, best_seq, best_fit, jpen, hpen, rpen = optimize_construct(df, linker, n_iter=800)

            st.success("Optimization complete!")

            st.markdown("### ✅ Optimized Construct")
            st.code(best_seq)
            st.write("Length:", len(best_seq), "aa")

            st.markdown("### 📊 Optimization Metrics")
            st.write("Fitness score:", round(best_fit, 4))
            st.write("Junction penalty:", round(jpen, 4))
            st.write("Hydrophobic clustering penalty:", round(hpen, 4))
            st.write("Repeat penalty:", round(rpen, 4))

            opt_df = df.set_index("Peptide").loc[best_order].reset_index()

            st.markdown("### 🖼️ Optimized Construct Architecture")
            fig = draw_vaccine_construct_figure(opt_df, linker=linker)
            st.pyplot(fig)

            st.session_state["optimized_construct"] = best_seq

# ---------------- TAB 4 ----------------
with tabs[3]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        fig,ax=plt.subplots(figsize=(12,4))
        ax.scatter(df["Start"],df["FinalScore"],c=df["FinalScore"],cmap="viridis",s=80)
        ax.set_title("Immunogenic Landscape")
        st.pyplot(fig)

# ---------------- TAB 5 ----------------
with tabs[4]:
    pdb_file = st.file_uploader("Upload PDB", type=["pdb"])
    if "df" in st.session_state and pdb_file:
        df=st.session_state["df"]
        pdb_text=pdb_file.read().decode("utf-8")
        style=st.selectbox("Style",["cartoon","surface","stick"])
        color_mode=st.selectbox("Color by",["score","cell","uniform"])
        sel=st.selectbox("Highlight",["ALL"]+df["Peptide"].tolist())
        show_structure_3d(pdb_text, df, sel, style, color_mode)

# ---------------- TAB 6 ----------------
with tabs[5]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        if st.button("Generate PyMOL script"):
            with open("highlight_epitopes.pml","w") as f:
                for _,r in df.iterrows():
                    s=r["Start"]; e=r["Start"]+r["Length"]
                    f.write(f"color red, resi {s}-{e}\n")
            st.success("Saved highlight_epitopes.pml")

# ---------------- TAB 7 ----------------
with tabs[6]:
    if "df" in st.session_state:
        df=st.session_state["df"]
        if st.button("Generate PDF"):
            fig,ax=plt.subplots(figsize=(10,4))
            ax.scatter(df["Start"],df["FinalScore"])
            with PdfPages("Epitope_Report.pdf") as pdf:
                pdf.savefig(fig)
            st.success("Saved Epitope_Report.pdf")

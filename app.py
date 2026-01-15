# =========================
# Unified Epitope Intelligence & Vaccine Design Platform
# CLEAN REBUILD — STEP 1 (Stable Core)
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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Unified Epitope Intelligence & Vaccine Design Platform",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("epitope_xgboost_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# =========================
# CONSTANTS
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

# =========================
# BASIC UTILITIES
# =========================
def read_fasta_multi(text):
    seqs = []
    cur = ""
    for l in text.strip().splitlines():
        if l.startswith(">"):
            if cur:
                seqs.append(cur)
                cur = ""
        else:
            cur += l.strip()
    if cur:
        seqs.append(cur)
    return [s.upper() for s in seqs]

def conservancy_percent(pep, seqs):
    return 100 * sum(pep in s for s in seqs) / len(seqs)

# =========================
# FEATURE EXTRACTION (ML)
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
    if total <= 0:
        return [0]*400
    return [counts[dp]/total for dp in dipeptides]

def physchem(seq):
    L = len(seq)
    mw = sum(aa_weights[a] for a in seq)
    hydv = sum(hydro[a] for a in seq) / L
    aromatic = sum(a in "FWY" for a in seq) / L
    return mw, hydv, aromatic

def extract_features(seq):
    aa = aa_composition(seq)
    dp = dipeptide_composition(seq)
    mw, hydv, aromatic = physchem(seq)
    return aa + dp + [len(seq), mw, hydv, aromatic]

# =========================
# UI
# =========================

st.title("🧬 Unified Epitope Intelligence & Vaccine Design Platform")

tabs = st.tabs([
    "Pipeline",
    "SHAP",
    "Vaccine",
    "Landscape",
    "3D",
    "Chemistry",
    "Export",
    "Report"
])

# =========================
# TAB 1 — PIPELINE
# =========================
with tabs[0]:
    st.header("🔬 Epitope Mining Pipeline")

    fasta_input = st.text_area("Paste FASTA sequences:")
    min_len = st.slider("Min length", 8, 15, 9)
    max_len = st.slider("Max length", 9, 25, 15)
    top_n = st.selectbox("Top N epitopes", [10,20,50,100])

    if st.button("Run Pipeline"):
        seqs = read_fasta_multi(fasta_input)

        if len(seqs) == 0:
            st.error("Please paste FASTA sequences.")
        else:
            main = seqs[0]

            peptides = []
            positions = []

            for L in range(min_len, max_len+1):
                for i in range(len(main)-L+1):
                    peptides.append(main[i:i+L])
                    positions.append(i+1)

            X = pd.DataFrame(
                [extract_features(p) for p in peptides],
                columns=feature_columns
            )

            probs = model.predict_proba(X)[:,1]

            rows = []
            for pep, pos, ml in zip(peptides, positions, probs):
                cons = conservancy_percent(pep, seqs)
                final = 0.7*ml + 0.3*(cons/100)
                rows.append([pep, pos, len(pep), ml, cons, final])

            df = pd.DataFrame(
                rows,
                columns=["Peptide","Start","Length","ML","Conservancy_%","FinalScore"]
            )

            df = df.sort_values("FinalScore", ascending=False).head(top_n)

            st.session_state["df"] = df
            st.session_state["X"] = X

            st.success("Pipeline completed.")
            st.dataframe(df)

# =========================
# TAB 2 — SHAP
# =========================
with tabs[1]:
    st.header("🧠 Model Explainability (SHAP)")
    if "df" in st.session_state:
        X = st.session_state["X"]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.iloc[:20])

        fig, _ = plt.subplots(figsize=(8,5), dpi=200)
        shap.summary_plot(shap_vals, X.iloc[:20], show=False)
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Run pipeline first.")

# =========================
# TAB 3 — VACCINE
# =========================
with tabs[2]:
    st.header("💉 Multi-Epitope Vaccine Designer")
    st.info("Vaccine design engine will appear here in STEP 3.")

# =========================
# TAB 4 — LANDSCAPE
# =========================
with tabs[3]:
    st.header("📊 Immunogenic Landscape")
    if "df" in st.session_state:
        df = st.session_state["df"]
        fig, ax = plt.subplots(figsize=(8,3), dpi=200)
        ax.scatter(df["Start"], df["FinalScore"], s=60)
        ax.set_title("Immunogenic Landscape")
        st.pyplot(fig, use_container_width=False)

# =========================
# TAB 5 — 3D
# =========================
with tabs[4]:
    st.header("🧬 3D Structure Viewer")
    pdb_file = st.file_uploader("Upload PDB file", type=["pdb"])
    if pdb_file:
        view = py3Dmol.view(width=900, height=600)
        view.addModel(pdb_file.read().decode("utf-8"), "pdb")
        view.setStyle({"cartoon":{"color":"lightgray"}})
        view.zoomTo()
        components.html(view._make_html(), height=650)

# =========================
# TAB 6 — CHEMISTRY
# =========================
with tabs[5]:
    st.header("🧪 Peptide Chemistry & Developability")
    st.info("Full chemistry engine will appear here in STEP 2.")

# =========================
# TAB 7 — EXPORT
# =========================
with tabs[6]:
    st.header("⬇️ Export Results")
    if "df" in st.session_state:
        df = st.session_state["df"]
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "epitopes.csv")

# =========================
# TAB 8 — REPORT
# =========================
with tabs[7]:
    st.header("📄 PDF Report")
    if "df" in st.session_state:
        if st.button("Generate PDF"):
            df = st.session_state["df"]
            fig, ax = plt.subplots(figsize=(6,4), dpi=200)
            ax.scatter(df["Start"], df["FinalScore"])
            with PdfPages("Report.pdf") as pdf:
                pdf.savefig(fig)

            with open("Report.pdf","rb") as f:
                st.download_button("Download PDF", f, "Epitope_Report.pdf")

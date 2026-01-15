
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
# CHEMISTRY ENGINE
# =========================

aliphatic = set("AVLIM")

def hydrophobic_moment(seq):
    L = len(seq)
    angles = [i*100*pi/180 for i in range(L)]
    mx = sum(hydro[seq[i]] * cos(angles[i]) for i in range(L))
    my = sum(hydro[seq[i]] * sin(angles[i]) for i in range(L))
    return (mx*mx + my*my)**0.5 / L

def gravy(seq):
    return sum(hydro[a] for a in seq) / len(seq)

def aliphatic_index(seq):
    return 100 * sum(a in aliphatic for a in seq) / len(seq)

def boman_index(seq):
    return -sum(hydro[a] for a in seq) / len(seq)

def aggregation_score(seq):
    return max(0, gravy(seq)) + sum(a in "IVLFWY" for a in seq)/len(seq)

def toxicity_score(seq):
    return (sum(a in "KR" for a in seq)/len(seq)) * gravy(seq)**2

def protease_stability(seq):
    cuts = sum(a in "KR" for a in seq[1:-1])
    return 1 / (1 + cuts)

def solubility_score(seq):
    return 1 / (1 + exp(gravy(seq)))

def membrane_binding_prob(seq):
    x = 0.8*hydrophobic_moment(seq) + 0.5*gravy(seq)
    return 1 / (1 + exp(-3*x))

def multi_objective_score(ml_score, seq):
    return 0.6*ml_score + 0.4*solubility_score(seq)

# =========================
# CHEMISTRY PLOTS (HIGH DPI)
# =========================

def plot_helical_wheel(seq):
    L = len(seq)
    angles = [i*100*pi/180 for i in range(L)]

    fig, ax = plt.subplots(figsize=(4,4), dpi=220)

    xs = []; ys = []
    for i,a in enumerate(angles):
        x = cos(a); y = sin(a)
        xs.append(x); ys.append(y)

        aa = seq[i]
        c = "red" if hydro[aa] > 1 else "blue" if hydro[aa] < -1 else "green"

        ax.scatter(x, y, s=600, c=c, alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.text(x, y, aa, ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    mx = sum(hydro[seq[i]] * xs[i] for i in range(L))
    my = sum(hydro[seq[i]] * ys[i] for i in range(L))
    ax.arrow(0, 0, mx/5, my/5, head_width=0.08, linewidth=2, color="black")

    circle = plt.Circle((0,0), 1, fill=False, linestyle="dashed", linewidth=1)
    ax.add_artist(circle)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Helical Wheel Projection", fontsize=12)

    return fig

def plot_hydropathy(seq):
    vals = [hydro[a] for a in seq]

    fig, ax = plt.subplots(figsize=(6,2.5), dpi=220)
    ax.plot(range(1, len(seq)+1), vals, marker="o", linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Hydropathy Plot", fontsize=11)
    ax.set_xlabel("Position")
    ax.set_ylabel("Hydrophobicity")

    return fig
# =========================
# VACCINE DESIGN CONSTANTS
# =========================

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
# CONSTRUCT QUALITY METRICS
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

def show_structure_3d(pdb_text, df):

    st.subheader("🧬 3D Structure Viewer (Performance Safe Mode)")

    # ============ UI ============
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        style = st.selectbox("Style", ["cartoon", "stick", "sphere", "line", "surface"])
    with c2:
        color_mode = st.selectbox("Color by", ["FinalScore", "Conservancy_%", "Electrostatic", "uniform"])
    with c3:
        focus = st.selectbox("Focus epitope", ["ALL"] + df["Peptide"].tolist())
    with c4:
        auto_rotate = st.checkbox("Auto rotate", value=False)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        opacity = st.slider("Transparency", 0.4, 1.0, 1.0)
    with c6:
        rot_speed = st.slider("Rotation speed", 0.1, 0.5, 0.2)
    with c7:
        bg = st.selectbox("Background", ["white", "black"])
    with c8:
        high_quality = st.checkbox("High quality (slow)", value=False)

    chain = st.text_input("Chain (leave empty for all):", "")

    # ============ Viewer ============
    view = py3Dmol.view(width=1000, height=650)
    view.addModel(pdb_text, "pdb")
    view.setBackgroundColor(bg)

    selector = {}
    if chain.strip():
        selector["chain"] = chain.strip()

    # ============ Base style ============
    if style == "surface":
        if not high_quality:
            st.warning("⚠ Surface is heavy. Enable High Quality only if needed.")
        view.setStyle(selector, {
            "surface": {
                "opacity": opacity,
                "colorscheme": "whiteCarbon",
                "resolution": 12 if high_quality else 6   # CRITICAL performance control
            }
        })
    else:
        view.setStyle(selector, {style: {"opacity": opacity}})

    # ============ Coloring ============
    scores = df["FinalScore"].values
    cons = df["Conservancy_%"].values
    smin, smax = scores.min(), scores.max()
    cmin, cmax = cons.min(), cons.max()

    def fast_color(t):
        # fast blue → green → red
        t = max(0, min(1, t))
        if t < 0.5:
            return f"rgb(0,{int(255*t*2)},255)"
        else:
            return f"rgb({int(255*(t-0.5)*2)},255,0)"

    for _, r in df.iterrows():
        pep = r["Peptide"]
        if focus != "ALL" and pep != focus:
            continue

        start = int(r["Start"])
        end = int(r["Start"] + r["Length"] - 1)

        if color_mode == "FinalScore":
            t = (r["FinalScore"] - smin) / (smax - smin + 1e-6)
        elif color_mode == "Conservancy_%":
            t = (r["Conservancy_%"] - cmin) / (cmax - cmin + 1e-6)
        elif color_mode == "Electrostatic":
            pepseq = r["Peptide"]
            charge_score = sum(1 if a in "KR" else -1 if a in "DE" else 0 for a in pepseq) / len(pepseq)
            t = (charge_score + 1) / 2
        else:
            t = 0.5

        col = fast_color(t)

        sel = {"resi": list(range(start, end+1))}
        if chain.strip():
            sel["chain"] = chain.strip()

        view.addStyle(sel, {
            "cartoon": {"color": col},
            "stick": {"color": col},
            "sphere": {"color": col},
            "line": {"color": col},
            "surface": {"color": col, "opacity": opacity}
        })

    # ============ Camera ============
    view.zoomTo()

    if auto_rotate:
        view.spin(True, rot_speed)

    # ============ Render ============
    components.html(view._make_html(), height=700, scrolling=False)

    st.info("💡 Tip: Use 'cartoon' or 'stick' for speed. Use 'surface' only when needed.")

def plot_immunogenic_landscape(df, color_by="FinalScore"):
    import numpy as np

    x = df["Start"].values
    y = df[color_by].values

    # Sort by position
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Smooth curve (simple moving average)
    window = max(3, len(y)//10)
    y_smooth = np.convolve(y, np.ones(window)/window, mode="same")

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200)

    # Scatter with color map
    sc = ax.scatter(
        x, y,
        c=y,
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.85
    )

    # Smoothed trend line
    ax.plot(x, y_smooth, color="red", linewidth=2, label="Smoothed trend")

    # Best epitope marker
    best_idx = np.argmax(y)
    ax.scatter(
        x[best_idx], y[best_idx],
        s=180,
        color="gold",
        edgecolors="black",
        zorder=5,
        label="Best epitope"
    )

    # Mean line
    mean_val = np.mean(y)
    ax.axhline(mean_val, linestyle="--", color="gray", linewidth=1, label="Mean")

    ax.set_title("Immunogenic Landscape Across Protein", fontsize=12)
    ax.set_xlabel("Position in protein")
    ax.set_ylabel(color_by)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_by)

    ax.legend()
    ax.grid(alpha=0.2)

    return fig

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
    st.header("💉 Intelligent Multi-Epitope Vaccine Designer")

    if "df" not in st.session_state:
        st.info("Run the pipeline first.")
    else:
        df = st.session_state["df"]

        colA, colB, colC = st.columns(3)

        with colA:
            sig = st.selectbox("Signal peptide", list(SIGNAL_PEPTIDES.keys()))
            adj = st.selectbox("Adjuvant", list(ADJUVANTS.keys()))
            linker = st.selectbox("Linker", ["GPGPG", "AAY", "EAAAK"])

        with colB:
            n_epi = st.slider("Number of epitopes", 3, min(20, len(df)), min(8, len(df)))
            cell_filter = st.selectbox("Epitope type", ["All"])  # placeholder for future

        with colC:
            ordering = st.selectbox("Order epitopes by", ["FinalScore", "Conservancy_%", "Start"])
            add_padre = st.checkbox("Add PADRE helper epitope", value=True)
            filter_bad = st.checkbox("Remove toxic / aggregating peptides", value=True)

        work = df.copy()

        if filter_bad:
            work = work[
                work["Peptide"].apply(
                    lambda p: aggregation_score(p) < 1.5 and toxicity_score(p) < 1
                )
            ]

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

        # =========================
        # VISUAL ARCHITECTURE
        # =========================
        st.subheader("🧱 Vaccine Architecture")

        cols = st.columns(len(blocks))
        for (label, seq), c in zip(blocks, cols):
            color = (
                "#4CAF50" if label=="Epitope"
                else "#FF9800" if label=="Adjuvant"
                else "#03A9F4" if label=="Signal"
                else "#9C27B0"
            )
            c.markdown(
                f"""
                <div style="padding:10px;border-radius:10px;background:{color};color:white;text-align:center">
                <b>{label}</b><br>{len(seq)} aa
                </div>
                """,
                unsafe_allow_html=True
            )

        # =========================
        # FINAL CONSTRUCT
        # =========================
        st.subheader("🧬 Final Vaccine Construct")
        st.code(construct)

        # =========================
        # CONSTRUCT METRICS
        # =========================
        st.subheader("📊 Construct Developability Metrics")

        qm = construct_quality_metrics(selected if len(selected) > 0 else None)

        if qm:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg GRAVY", f"{qm['Avg GRAVY']:.2f}")
            c2.metric("Avg Solubility", f"{qm['Avg Solubility']:.2f}")
            c3.metric("Avg Aggregation", f"{qm['Avg Aggregation']:.2f}")
            c4.metric("Avg Membrane Bind", f"{qm['Avg Membrane Binding']:.2f}")
            c5.metric("Developability", f"{qm['Developability Score']:.2f}")

        # =========================
        # EXPORT
        # =========================
        st.subheader("⬇️ Export Vaccine")

        fasta = f">Multi_epitope_vaccine\n{construct}"
        st.download_button("Download FASTA", fasta, "vaccine.fasta")
        st.download_button("Download TXT", construct, "vaccine.txt")

# =========================
# LANDSCAPE (ADVANCED)
# =========================
with tabs[3]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("📊 Immunogenic Landscape Analysis")

        color_by = st.selectbox(
            "Color / plot by",
            ["FinalScore", "Conservancy_%", "Antigenicity"]
        )

        fig = plot_immunogenic_landscape(df, color_by=color_by)
        st.pyplot(fig, use_container_width=False)

        st.info("🔍 This plot shows immunogenic hotspots, clustering, and overall coverage along the protein.")

# =========================
# TAB 5 — 3D
# =========================
with tabs[4]:
    st.header("🧬 3D Structure Viewer")
    pdb_file = st.file_uploader("Upload PDB file", type=["pdb"])

    if "df" not in st.session_state:
        st.info("Run the pipeline first.")
    elif pdb_file:
        show_structure_3d(pdb_file.read().decode("utf-8"), st.session_state["df"])

# =========================
# TAB 6 — CHEMISTRY
# =========================
with tabs[5]:
    st.header("🧪 Peptide Chemistry & Developability")

    if "df" not in st.session_state:
        st.info("Run the pipeline first.")
    else:
        df = st.session_state["df"]

        pep = st.selectbox("Select peptide", df["Peptide"])
        ml_score = df[df["Peptide"] == pep]["FinalScore"].values[0]

        st.subheader("🧬 Sequence")
        st.code(pep)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("GRAVY", f"{gravy(pep):.2f}")
        col2.metric("Hydrophobic Moment", f"{hydrophobic_moment(pep):.3f}")
        col3.metric("Aliphatic Index", f"{aliphatic_index(pep):.1f}")
        col4.metric("Boman Index", f"{boman_index(pep):.2f}")

        col1.metric("Membrane Binding", f"{membrane_binding_prob(pep):.2f}")
        col2.metric("Solubility", f"{solubility_score(pep):.2f}")
        col3.metric("Aggregation Risk", f"{aggregation_score(pep):.2f}")
        col4.metric("Toxicity Risk", f"{toxicity_score(pep):.2f}")

        col1.metric("Protease Stability", f"{protease_stability(pep):.2f}")
        col2.metric("Multi-objective Score", f"{multi_objective_score(ml_score, pep):.3f}")

        st.subheader("🌀 Helical Wheel")
        st.pyplot(plot_helical_wheel(pep), use_container_width=False)

        st.subheader("📈 Hydropathy Plot")
        st.pyplot(plot_hydropathy(pep), use_container_width=False)

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

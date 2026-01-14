import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
df = df_hits.copy()   # or df if you want all peptides
score_col = "Score" if "Score" in df.columns else "FinalScore"

protein_length = int(df["Start_Position"].max() + df["Length"].max())

st.markdown("## 🧬 Advanced Visual Analytics Dashboard")

# ======================================================
# 1️⃣ MULTI-TRACK GENOME-BROWSER STYLE EPITOPE MAP
# ======================================================
st.subheader("🧬 Multi-track Epitope Landscape Map")

fig = go.Figure()

# Protein backbone
fig.add_trace(go.Scatter(
    x=[1, protein_length],
    y=[0, 0],
    mode="lines",
    line=dict(width=10, color="black"),
    name="Protein"
))

# Epitopes
for _, r in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[r.Start_Position, r.Start_Position + r.Length],
        y=[1, 1],
        mode="lines",
        line=dict(width=12),
        hovertext=f"""
Peptide: {r.Peptide}
Score: {r[score_col]:.3f}
Cell: {r.Cell_Type}
Conservancy: {r['Conservancy_%']:.1f}%
"""
    ))

fig.update_layout(
    title="Epitope Mapping Along Protein Sequence",
    xaxis_title="Protein Position",
    yaxis=dict(visible=False),
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2️⃣ COVERAGE MAP (PER-RESIDUE EPITOPE DENSITY)
# ======================================================
st.subheader("📍 Epitope Coverage Map (Per-Residue Density)")

coverage = np.zeros(protein_length)

for _, r in df.iterrows():
    s = int(r.Start_Position) - 1
    e = int(r.Start_Position + r.Length) - 1
    coverage[s:e] += 1

cov_df = pd.DataFrame({
    "Position": np.arange(1, protein_length + 1),
    "Coverage": coverage
})

fig_cov = px.line(
    cov_df, x="Position", y="Coverage",
    title="Epitope Coverage Density Along Protein"
)
fig_cov.update_layout(template="plotly_white")

st.plotly_chart(fig_cov, use_container_width=True)

# ======================================================
# 3️⃣ EPITOPE CLUSTERING HEATMAP
# ======================================================
st.subheader("🧠 Epitope Clustering Heatmap")

X = df[["Start_Position", "Length", score_col, "Conservancy_%"]].values

k = min(4, len(df))
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters

heat_df = df.sort_values("Cluster")[["Start_Position", "Length", score_col, "Conservancy_%"]]

fig_heat = px.imshow(
    heat_df.T,
    aspect="auto",
    color_continuous_scale="Viridis",
    title="Epitope Feature Clustering Heatmap"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ======================================================
# 4️⃣ MODEL CONFIDENCE VIOLIN PLOT
# ======================================================
st.subheader("🎻 Model Confidence Distribution")

fig_violin = px.violin(
    df,
    y=score_col,
    x="Cell_Type",
    box=True,
    points="all",
    color="Cell_Type",
    title="Model Confidence Distribution by Epitope Type"
)

fig_violin.update_layout(template="plotly_white")

st.plotly_chart(fig_violin, use_container_width=True)

# ======================================================
# 5️⃣ SHAP EXPLAINABILITY (OPTIONAL, SAFE)
# ======================================================
st.subheader("🧠 Model Explainability (SHAP Summary)")

try:
    import shap

    # Use a small subset for speed
    X_shap = X[: min(200, len(X))]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_shap)

    shap_df = pd.DataFrame(np.abs(shap_values.values).mean(axis=0)).T

    fig_shap = px.bar(
        shap_df.T,
        title="Mean |SHAP| Feature Importance",
        labels={"index":"Feature Index","value":"Importance"}
    )
    fig_shap.update_layout(template="plotly_white")

    st.plotly_chart(fig_shap, use_container_width=True)

except Exception as e:
    st.info("SHAP not available in this environment. (This is optional and does not affect main results.)")


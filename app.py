import plotly.express as px
import plotly.graph_objects as go

st.markdown("## 📊 Advanced Interactive Analysis Dashboard")

df = df_hits.copy()

# ===============================
# 1️⃣ Epitope Length Distribution
# ===============================
st.subheader("📏 Epitope Length Distribution")

len_df = df["Length"].value_counts().sort_index().reset_index()
len_df.columns = ["Length", "Count"]

fig1 = px.bar(
    len_df, x="Length", y="Count",
    color="Count",
    color_continuous_scale="viridis",
    title="Distribution of Epitope Lengths"
)
fig1.update_layout(template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# ===============================
# 2️⃣ Cell Type Distribution
# ===============================
st.subheader("🧬 B-cell vs T-cell Distribution")

cell_df = df["Cell_Type"].value_counts().reset_index()
cell_df.columns = ["Cell_Type", "Count"]

fig2 = px.pie(
    cell_df,
    names="Cell_Type",
    values="Count",
    hole=0.45,
    title="B-cell vs T-cell Epitope Proportion"
)
st.plotly_chart(fig2, use_container_width=True)

# ===============================
# 3️⃣ Toxicity Risk Profile
# ===============================
st.subheader("☣️ Toxicity Risk Profile")

tox_df = df["Toxicity_Risk"].value_counts().reset_index()
tox_df.columns = ["Risk", "Count"]

fig3 = px.bar(
    tox_df, x="Risk", y="Count",
    color="Risk",
    color_discrete_map={"Low":"green","High":"red"},
    title="Toxicity Risk Distribution"
)
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# ===============================
# 4️⃣ Conservancy Distribution
# ===============================
st.subheader("🧪 Conservancy Percentage Distribution")

fig4 = px.histogram(
    df,
    x="Conservancy_%",
    nbins=10,
    title="Conservancy Score Distribution",
    color_discrete_sequence=["#636EFA"]
)
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# ===============================
# 5️⃣ Epitope Hotspot Landscape
# ===============================
st.subheader("📍 Epitope Hotspot Map Along Protein")

fig5 = px.scatter(
    df,
    x="Start_Position",
    y="Score",
    size="Length",
    color="Score",
    hover_data=["Peptide","Length","Conservancy_%"],
    color_continuous_scale="Turbo",
    title="Epitope Hotspot Landscape Across Protein"
)

fig5.update_layout(
    xaxis_title="Protein Position",
    yaxis_title="Epitope Score",
    template="plotly_white"
)

st.plotly_chart(fig5, use_container_width=True)

# ===============================
# 6️⃣ Screening Funnel View
# ===============================
st.subheader("🧹 Screening Funnel Overview")

funnel_df = pd.DataFrame({
    "Stage": ["Predicted", "After Toxicity", "After Allergenicity", "Final Selected"],
    "Count": [
        len(df_hits),
        len(df_hits[df_hits["Toxicity_Risk"]=="Low"]),
        len(df_hits[(df_hits["Toxicity_Risk"]=="Low") & (df_hits["Allergenicity_Risk"]=="Low")]),
        len(df_hits)
    ]
})

fig6 = px.funnel(
    funnel_df,
    x="Count",
    y="Stage",
    title="Epitope Screening Funnel"
)

st.plotly_chart(fig6, use_container_width=True)

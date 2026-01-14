st.subheader("📊 Advanced Epitope Analysis Dashboard")

# =========================
# 1) MULTI-TRACK GENOME BROWSER STYLE PLOT
# =========================
st.markdown("## 🧬 Multi-track Epitope Landscape")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Track 1: Epitope density
bins = np.arange(1, df["Start"].max()+20, 10)
hist, edges = np.histogram(df["Start"], bins=bins)
axes[0].bar(edges[:-1], hist, width=10, color="steelblue")
axes[0].set_ylabel("Density")
axes[0].set_title("Epitope Density")

# Track 2: Final Score landscape
axes[1].scatter(df["Start"], df["FinalScore"], c=df["FinalScore"], cmap="viridis")
axes[1].set_ylabel("Final Score")
axes[1].set_title("Final Score Landscape")

# Track 3: Conservancy
axes[2].scatter(df["Start"], df["Conservancy_%"], c=df["Conservancy_%"], cmap="plasma")
axes[2].set_ylabel("Conservancy %")
axes[2].set_title("Conservancy Track")

# Track 4: Screening flags
tox_map = df["Toxicity"].map({"Low":0, "High":1})
all_map = df["Allergenicity"].map({"Low":0, "High":1})
axes[3].scatter(df["Start"], tox_map, label="Toxicity", color="red", alpha=0.6)
axes[3].scatter(df["Start"], all_map+1.2, label="Allergenicity", color="black", alpha=0.6)
axes[3].set_yticks([0,1,2,3])
axes[3].set_yticklabels(["Tox Low","Tox High","All Low","All High"])
axes[3].set_title("Screening Flags")
axes[3].legend()

axes[3].set_xlabel("Protein Position")

plt.tight_layout()
st.pyplot(fig)

# =========================
# 2) EPITOPE CLUSTERING HEATMAP
# =========================
st.markdown("## 🧪 Epitope Feature Clustering")

heat_df = df[["FinalScore","Conservancy_%","Antigenicity","Length"]]
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(heat_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

# =========================
# 3) SCORE DISTRIBUTION (VIOLIN)
# =========================
st.markdown("## 🎻 Model Confidence Distribution")

fig3, ax3 = plt.subplots(figsize=(8,4))
sns.violinplot(y=df["FinalScore"], ax=ax3, color="lightgreen")
ax3.set_title("Final Score Distribution")
st.pyplot(fig3)

# =========================
# 4) CELL TYPE DONUT
# =========================
st.markdown("## 🧬 Cell Type Composition")

cell_counts = df["Cell_Type"].value_counts()
fig4, ax4 = plt.subplots(figsize=(6,6))
ax4.pie(cell_counts, labels=cell_counts.index, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
ax4.set_title("B-cell vs T-cell vs Both")
st.pyplot(fig4)

# =========================
# 5) SCREENING FUNNEL
# =========================
st.markdown("## 🧹 Screening Funnel")

total = len(rows)
non_tox = sum(r[7]=="Low" for r in rows)
non_all = sum((r[7]=="Low" and r[8]=="Low") for r in rows)
final = len(df)

funnel_df = pd.DataFrame({
    "Stage": ["All", "Non-Toxic", "Non-Allergenic", "Final"],
    "Count": [total, non_tox, non_all, final]
})

fig5, ax5 = plt.subplots(figsize=(8,4))
ax5.barh(funnel_df["Stage"], funnel_df["Count"], color=["gray","orange","green","blue"])
ax5.set_title("Epitope Screening Funnel")
st.pyplot(fig5)

# =========================
# 6) HOTSPOT MAP
# =========================
st.markdown("## 📍 Epitope Hotspot Map")

fig6, ax6 = plt.subplots(figsize=(12,4))
sc = ax6.scatter(df["Start"], df["FinalScore"], s=df["Length"]*20, c=df["FinalScore"], cmap="turbo")
ax6.set_xlabel("Protein Position")
ax6.set_ylabel("Final Score")
ax6.set_title("Epitope Hotspots Along Protein")
plt.colorbar(sc, ax=ax6, label="Final Score")
st.pyplot(fig6)

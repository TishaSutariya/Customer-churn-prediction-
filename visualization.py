import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load data
print("Loading data...")
df = pd.read_csv("data/processed/cleaned_churn_data.csv")
print(f"Loaded {len(df)} customers")
# Load model
model = joblib.load("models/trained/churn_model.pkl")
# 1. Churn Distribution Pie Chart
print("\n1. Creating Churn Distribution Pie Chart...")
churn_counts = df["Churn"].value_counts()
labels = ["No Churn", "Churn"]
values = [churn_counts[0], churn_counts[1]]

fig_pie = px.pie(
    values=values,
    names=labels,
    hole=0.4,
    color_discrete_sequence=["#2ecc71", "#3c75e7"],
    title="Customer Churn Distribution"
)

fig_pie.update_traces(texttemplate="%{percent:.1f}%")
fig_pie.write_image("reports/churn_distribution.png", width=800, height=600)
print("Saved to reports/churn_distribution.png")

# 2. Churn by Contract Bar Chart
print("\n2. Creating Churn by Contract Bar Chart...")
df_temp = df.copy()
df_temp["Contract Name"] = df_temp["Contract"].map({
    0: "Month-to-month",
    1: "One year",
    2: "Two year"
})

churn_by_contract = df_temp.groupby("Contract Name", observed=False)["Churn"].mean().reset_index()
churn_by_contract["Churn Rate %"] = churn_by_contract["Churn"] * 100

fig_contract = px.bar(
    churn_by_contract,
    x="Contract Name",
    y="Churn Rate %",
    text="Churn Rate %",
    color="Churn Rate %",
    color_continuous_scale="Reds",
    title="Churn Rate by Contract Type"
)
fig_contract.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_contract.update_layout(xaxis_title="Contract Type", yaxis_title="Churn Rate (%)")
fig_contract.write_image("reports/churn_by_contract.png", width=800, height=600)
print("Saved to reports/churn_by_contract.png")

# 3. Churn by Tenure Bar Chart
print("\n3. Creating Churn by Tenure Bar Chart...")

df_temp["Tenure Group"] = pd.cut(
    df_temp["tenure"],
    bins=[0, 6, 12, 24, 48, 100],
    labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
)

churn_by_tenure = df_temp.groupby("Tenure Group", observed=False)["Churn"].mean().reset_index()
churn_by_tenure["Churn Rate %"] = churn_by_tenure["Churn"] * 100

fig_tenure = px.bar(
    churn_by_tenure,
    x="Tenure Group",
    y="Churn Rate %",
    color="Churn Rate %",
    color_continuous_scale="OrRd",
    title="Churn Rate by Tenure Group"
)

fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_tenure.update_layout(xaxis_title="Tenure Group", yaxis_title="Churn Rate (%)")
fig_tenure.write_image("reports/churn_by_tenure.png", width=800, height=600)
print("Saved to reports/churn_by_tenure.png")

# 4. Monthly Charges Box Plot
print("\n4. Creating Monthly Charges Box Plot...")

fig_charges = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges vs Churn",
    labels={"Churn": "Churn (0=No, 1=Yes)", "MonthlyCharges": "Monthly Charges ($)"}
)

fig_charges.update_layout(showlegend=True)
fig_charges.write_image("reports/churn_by_charges.png", width=800, height=600)
print("Saved to reports/churn_by_charges.png")

# 5. Feature Importance Bar Chart
print("\n5. Creating Feature Importance Chart...")

importance_df = pd.read_csv("reports/feature_importance.csv")
top_10 = importance_df.head(10).copy()
top_10["Importance %"] = top_10["Importance"] * 100

fig_importance = px.bar(
    top_10,
    x="Importance %",
    y="Feature",
    orientation="h",
    color="Importance %",
    color_continuous_scale="Blues",
    title="Top 10 Churn Drivers"
)

fig_importance.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_importance.update_layout(xaxis_title="Importance (%)", yaxis_title="", showlegend=False)
fig_importance.write_image("reports/feature_importance.png", width=800, height=600)
print("Saved to reports/feature_importance.png")

# 6. Churn by Internet Service
print("\n6. Creating Churn by Internet Service Chart...")

churn_by_internet = df.groupby(["InternetService", "Churn"], observed=False).size().reset_index(name="Count")
churn_by_internet["Internet Name"] = churn_by_internet["InternetService"].map({
    0: "No",
    1: "DSL",
    2: "Fiber optic"
})

fig_internet = px.sunburst(
    churn_by_internet,
    path=["Internet Name", "Churn"],
    values="Count",
    title="Churn by Internet Service Type"
)

fig_internet.write_image("reports/churn_by_internet.png", width=800, height=600)
print("Saved to reports/churn_by_internet.png")

# 7. Correlation Heatmap
print("\n7. Creating Correlation Heatmap...")

numeric_df = df.select_dtypes(include=[float, int])
corr = numeric_df.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Feature Correlation Matrix",
    color_continuous_scale="RdBu_r"
)

fig_corr.write_image("reports/correlation_matrix.png", width=800, height=600)
print("Saved to reports/correlation_matrix.png")

# Summary
print("\n" + "=" * 60)
print("ALL VISUALIZATIONS CREATED!")
print("=" * 60)
print("\n7 PNG files saved in reports/ folder:")
print("  1. churn_distribution.png")
print("  2. churn_by_contract.png")
print("  3. churn_by_tenure.png")
print("  4. churn_by_charges.png")
print("  5. feature_importance.png")
print("  6. churn_by_internet.png")
print("  7. correlation_matrix.png")
print("\nUse these PNG files in reports/presentations")
print("=" * 60)
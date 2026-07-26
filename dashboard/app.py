import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn_data.csv"
MODEL_PATH = BASE_DIR / "models" / "trained" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "trained" / "scaler.pkl"
REPORT_PATH = BASE_DIR / "reports"
BANNER_PATH = BASE_DIR / "banner.png"

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Dashboard")
st.image(str(BANNER_PATH), use_container_width=True)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Predict Churn",
        "Model Performance",
        "Analytics",
        "Charts"
    ]
)

if page == "Home":

    st.header("Welcome")

    st.write("""
This dashboard can be used to:

- Predict customer churn
- View model performance
- Explore customer analytics
- View important charts
""")

elif page == "Predict Churn":

    st.header("Predict Customer Churn")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_clean = pd.read_csv(DATA_PATH)
    feature_columns = df_clean.drop("Churn", axis=1).columns

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    tenure = st.slider("Tenure", 1, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    internet = st.selectbox(
        "Internet Service",
        ["No", "DSL", "Fiber optic"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    monthly = st.slider(
        "Monthly Charges",
        20.0,
        120.0,
        70.0
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Credit card",
            "Electronic check",
            "Mailed check"
        ]
    )

    if st.button("Predict"):

        customer = {
            "gender": 1 if gender == "Female" else 0,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 0,
            "tenure": tenure,
            "PhoneService": 1 if phone == "Yes" else 0,
            "MultipleLines": 0,
            "InternetService": {
                "No":0,
                "DSL":1,
                "Fiber optic":2
            }[internet],
            "OnlineSecurity":0,
            "OnlineBackup":0,
            "DeviceProtection":0,
            "TechSupport":0,
            "StreamingTV":0,
            "StreamingMovies":0,
            "Contract":{
                "Month-to-month":0,
                "One year":1,
                "Two year":2
            }[contract],
            "PaperlessBilling":0,
            "MonthlyCharges":monthly,
            "TotalCharges":monthly*tenure,
            "PaymentMethod_Credit card (automatic)":1 if payment=="Credit card" else 0,
            "PaymentMethod_Electronic check":1 if payment=="Electronic check" else 0,
            "PaymentMethod_Mailed check":1 if payment=="Mailed check" else 0,
            "PaymentMethod_Bank transfer (automatic)":0
        }

        df = pd.DataFrame([customer])

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        X = scaler.transform(df)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        if prediction == 1:
            result = "Customer Will Churn"
        else:
            result = "Customer Will Stay"

        st.subheader(result)
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

        if probability >= 0.70:
            st.error("High Risk Customer")
        elif probability >= 0.30:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")

elif page == "Model Performance":

    st.header("Model Performance")

    accuracy = 0.7722
    precision = 0.5570
    recall = 0.6925
    f1 = 0.6174
    roc = 0.8324

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy*100:.2f}%")
    c2.metric("Precision", f"{precision*100:.2f}%")
    c3.metric("Recall", f"{recall*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")
    c5.metric("ROC AUC", f"{roc*100:.2f}%")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Score": [accuracy, precision, recall, f1, roc]
    })

    fig = px.bar(
        metrics,
        x="Metric",
        y="Score",
        color="Metric",
        text="Score"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)


elif page == "Analytics":

    st.header("Analytics")

    df = pd.read_csv(DATA_PATH)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", len(df))
    c2.metric("Features", df.shape[1]-1)
    c3.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
    c4.metric("Average Tenure", f"{df['tenure'].mean():.0f} Months")

    st.subheader("Churn Distribution")

    churn = df["Churn"].value_counts()

    fig = px.pie(
        values=[churn[0], churn[1]],
        names=["No Churn", "Churn"],
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contract Wise Churn")

    contract = (
        df.groupby("Contract")["Churn"]
        .mean()
        .reset_index()
    )

    contract["Contract"] = contract["Contract"].map({
        0: "Month-to-month",
        1: "One Year",
        2: "Two Year"
    })

    contract["Churn Rate"] = contract["Churn"] * 100

    fig = px.bar(
        contract,
        x="Contract",
        y="Churn Rate",
        color="Contract",
        text="Churn Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")

    st.write("- Month-to-month customers churn the most.")
    st.write("- Long-term customers are more loyal.")
    st.write("- High monthly charges increase churn.")
    st.write("- Fiber optic users have higher churn.")
    st.write("- Tenure is an important factor.")


elif page == "Charts":

    st.header("Charts")

    df = pd.read_csv(DATA_PATH)

    st.subheader("Monthly Charges vs Churn")

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Important Features")

    importance = pd.read_csv(REPORT_PATH / "feature_importance.csv")

    top10 = importance.head(10)

    top10["Importance"] = top10["Importance"] * 100

    fig = px.bar(
        top10,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        text="Importance"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")

    corr = df.select_dtypes(include=["int64", "float64"]).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)
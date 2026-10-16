# ============================================================
# CUSTOMER CHURN INTELLIGENCE DASHBOARD
# Power BI-Inspired Full-Width Streamlit Application
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_churn_data.csv"
)


# ============================================================
# 3. THEME AND CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background-color: #F3F6FA;
        color: #243B53;
    }

    .block-container {
        max-width: 100%;
        padding: 1.2rem 1.8rem 2rem 1.8rem;
    }

    /* Hide sidebar */

    section[data-testid="stSidebar"] {
        display: none;
    }

    button[data-testid="collapsedControl"] {
        display: none;
    }

    /* Header */

    .dashboard-header {
        background: linear-gradient(
            110deg,
            #1F344D,
            #294C6B
        );
        padding: 23px 28px;
        border-radius: 12px;
        margin-bottom: 18px;
        color: #FFFFFF;
    }

    .dashboard-title {
        font-size: 29px;
        font-weight: 750;
        color: #FFFFFF;
        margin: 0;
    }

    .dashboard-subtitle {
        font-size: 12px;
        color: #D9E5F0;
        margin-top: 7px;
    }

    /* Section headings */

    .section-heading {
        font-size: 20px;
        font-weight: 750;
        color: #243B53;
        margin-top: 15px;
        margin-bottom: 9px;
    }

    .section-subtitle {
        color: #627D98;
        font-size: 12px;
        margin-bottom: 12px;
    }

    /* KPI cards */

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: none;
    }

    div[data-testid="stMetricLabel"] {
        color: #627D98;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #243B53;
        font-size: 23px;
        font-weight: 750;
    }

    /* Forms */

    div[data-testid="stForm"] {
        background: #FFFFFF;
        padding: 16px;
        border: 1px solid #DCE5EF;
        border-radius: 10px;
    }

    /* Containers */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 10px;
    }

    /* Buttons */

    .stButton > button {
        background: #294C6B;
        color: #FFFFFF;
        border-radius: 7px;
        border: 1px solid #294C6B;
        font-weight: 650;
    }

    .stButton > button:hover {
        background: #1F344D;
        color: #FFFFFF;
        border-color: #1F344D;
    }

    .stDownloadButton > button {
        border-radius: 7px;
        font-weight: 600;
    }

    /* Dividers */

    hr {
        border-color: #DCE5EF;
        margin: 15px 0;
    }

    /* Smaller chart headings */

    h4 {
        color: #294C6B !important;
        font-size: 15px !important;
    }

    /* Data table */

    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE5EF;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower(),
    )


def find_column(dataframe, possible_names):

    normalized_columns = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    for name in possible_names:

        key = normalize_name(name)

        if key in normalized_columns:
            return normalized_columns[key]

    return None


def format_currency(value):

    if pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


def make_chart(fig, height=275):

    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            color="#334155",
            size=10,
        ),
        margin=dict(
            l=12,
            r=12,
            t=38,
            b=18,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=9),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E8EEF5",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E8EEF5",
        zeroline=False,
    )

    return fig


def convert_churn_target(series):

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce",
        )

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "churned": 1,
        "retained": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

@st.cache_data
def load_dataset(path, modification_time):

    del modification_time

    return pd.read_csv(path)


if not DATA_PATH.exists():

    st.error(
        "Cleaned dataset not found."
    )

    st.code(str(DATA_PATH))

    st.info(
        "Check data/processed/cleaned_churn_data.csv."
    )

    st.stop()


try:

    df = load_dataset(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

except Exception as error:

    st.error(
        f"Unable to load the dataset: {error}"
    )

    st.stop()


df.columns = [
    str(column).strip()
    for column in df.columns
]


# ============================================================
# 6. PREPARE DATA
# ============================================================

churn_col = find_column(
    df,
    ["Churn", "Customer Churn", "Exited"],
)

if churn_col is None:

    st.error(
        "Churn column was not found."
    )

    st.write(
        "Available columns:",
        df.columns.tolist(),
    )

    st.stop()


df[churn_col] = convert_churn_target(
    df[churn_col]
)

df = df.dropna(
    subset=[churn_col]
).copy()

df[churn_col] = df[churn_col].astype(int)

df = df[
    df[churn_col].isin([0, 1])
].copy()


contract_col = find_column(
    df,
    ["Contract", "Contract Type"],
)

internet_col = find_column(
    df,
    ["InternetService", "Internet Service"],
)

tenure_col = find_column(
    df,
    ["tenure", "Tenure"],
)

monthly_col = find_column(
    df,
    ["MonthlyCharges", "Monthly Charges"],
)

total_col = find_column(
    df,
    ["TotalCharges", "Total Charges"],
)

customer_id_col = find_column(
    df,
    ["customerID", "CustomerID", "Customer ID"],
)


# ============================================================
# 7. PREPARE MODEL FEATURES
# ============================================================

model_df = df.copy()

drop_columns = [churn_col]

if customer_id_col:
    drop_columns.append(customer_id_col)

X = model_df.drop(
    columns=drop_columns,
    errors="ignore",
)

y = model_df[churn_col].astype(int)

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int,
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

X = X.loc[
    :,
    X.nunique(dropna=False) > 1
]

X = X.dropna(
    axis=1,
    how="all",
)

if X.empty:
    st.error("No usable model features were found.")
    st.stop()

if y.nunique() != 2:
    st.error("The dataset must contain both churn classes.")
    st.stop()


# ============================================================
# 8. MODEL TRAINING AND COMPARISON
# ============================================================

@st.cache_resource
def train_models(X_data, y_data):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_data,
            y_data,
            test_size=0.20,
            random_state=42,
            stratify=y_data,
        )
    )

    definitions = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "Decision Tree":
            __import__(
                "sklearn.tree",
                fromlist=["DecisionTreeClassifier"],
            ).DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
            ),
    }

    models = {}
    metrics = []
    predictions = {}
    probabilities = {}

    for name, estimator in definitions.items():

        pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predicted = pipeline.predict(
            X_test,
        )

        probability = pipeline.predict_proba(
            X_test,
        )[:, 1]

        try:

            auc = roc_auc_score(
                y_test,
                probability,
            )

        except ValueError:

            auc = np.nan

        metrics.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(
                    y_test,
                    predicted,
                ),
                "Precision": precision_score(
                    y_test,
                    predicted,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predicted,
                    zero_division=0,
                ),
                "F1 Score": f1_score(
                    y_test,
                    predicted,
                    zero_division=0,
                ),
                "ROC-AUC": auc,
            }
        )

        models[name] = pipeline
        predictions[name] = predicted
        probabilities[name] = probability

    metrics_df = pd.DataFrame(metrics)

    metrics_df = metrics_df.sort_values(
        "ROC-AUC",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return (
        models,
        metrics_df,
        X_test,
        y_test,
        predictions,
        probabilities,
    )


with st.spinner(
    "Preparing analytics and training comparison models..."
):

    (
        trained_models,
        model_metrics,
        X_test,
        y_test,
        model_predictions,
        model_probabilities,
    ) = train_models(
        X,
        y,
    )


# ============================================================
# 9. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">
            Customer Churn Intelligence
        </div>
        <div class="dashboard-subtitle">
            AI-Powered Customer Insights | Predictive Analytics | Retention Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. FILTERS
# ============================================================

st.markdown(
    '<div class="section-heading">Dashboard Filters</div>',
    unsafe_allow_html=True,
)

filtered_df = df.copy()

filter1, filter2, filter3, filter4 = st.columns(4)

with filter1:

    if contract_col:

        contracts = sorted(
            df[contract_col].dropna().unique().tolist(),
            key=str,
        )

        selected_contracts = st.multiselect(
            "Contract",
            contracts,
            default=contracts,
        )

        filtered_df = filtered_df[
            filtered_df[contract_col].isin(
                selected_contracts
            )
        ]


with filter2:

    if internet_col:

        internet_options = sorted(
            df[internet_col].dropna().unique().tolist(),
            key=str,
        )

        selected_internet = st.multiselect(
            "Internet Service",
            internet_options,
            default=internet_options,
        )

        filtered_df = filtered_df[
            filtered_df[internet_col].isin(
                selected_internet
            )
        ]


with filter3:

    if tenure_col:

        tenure_values = pd.to_numeric(
            df[tenure_col],
            errors="coerce",
        ).dropna()

        if not tenure_values.empty:

            low = int(tenure_values.min())
            high = int(tenure_values.max())

            if low < high:

                tenure_range = st.slider(
                    "Tenure (Months)",
                    min_value=low,
                    max_value=high,
                    value=(low, high),
                )

                filtered_df = filtered_df[
                    pd.to_numeric(
                        filtered_df[tenure_col],
                        errors="coerce",
                    ).between(
                        tenure_range[0],
                        tenure_range[1],
                    )
                ]


with filter4:

    status_filter = st.selectbox(
        "Customer Status",
        [
            "All Customers",
            "Churned",
            "Retained",
        ],
    )

    if status_filter == "Churned":

        filtered_df = filtered_df[
            filtered_df[churn_col] == 1
        ]

    elif status_filter == "Retained":

        filtered_df = filtered_df[
            filtered_df[churn_col] == 0
        ]


if filtered_df.empty:

    st.warning(
        "No customers match these filters. "
        "Please change your selection."
    )

    st.stop()


# ============================================================
# 11. BUSINESS OVERVIEW KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-heading">Business Overview</div>',
    unsafe_allow_html=True,
)

total_customers = len(filtered_df)

churned_customers = int(
    (filtered_df[churn_col] == 1).sum()
)

retained_customers = int(
    (filtered_df[churn_col] == 0).sum()
)

churn_rate = (
    churned_customers / total_customers * 100
)

retention_rate = (
    retained_customers / total_customers * 100
)

average_charges = (
    pd.to_numeric(
        filtered_df[monthly_col],
        errors="coerce",
    ).mean()
    if monthly_col
    else np.nan
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

with k2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%",
    )

with k3:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}",
    )

with k4:
    st.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%",
    )

with k5:
    st.metric(
        "Avg. Monthly Charges",
        format_currency(average_charges),
    )


# ============================================================
# 12. CUSTOMER PREDICTION SECTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">Customer Churn Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        Enter customer information to estimate churn probability.
    </div>
    """,
    unsafe_allow_html=True,
)


def match_feature(feature_columns, aliases):

    normalized = {
        normalize_name(column): column
        for column in feature_columns
    }

    for alias in aliases:

        key = normalize_name(alias)

        if key in normalized:
            return normalized[key]

    return None


def make_prediction_row(
    feature_columns,
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone,
    multiple_lines,
    internet,
    security,
    backup,
    device,
    support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless,
    payment,
    monthly_charges,
):

    row = pd.DataFrame(
        [
            {
                column: 0
                for column in feature_columns
            }
        ]
    ).astype(float)

    medians = X.median(
        numeric_only=True,
    )

    # Set sensible defaults for remaining features.

    for column in feature_columns:

        if column in medians.index:
            row.loc[0, column] = medians[column]

    def set_value(aliases, value):

        column = match_feature(
            feature_columns,
            aliases,
        )

        if column is not None:
            row.loc[0, column] = value

    # Demographics

    set_value(
        ["gender", "Gender"],
        1 if gender == "Female" else 0,
    )

    set_value(
        ["SeniorCitizen"],
        1 if senior == "Yes" else 0,
    )

    set_value(
        ["Partner"],
        1 if partner == "Yes" else 0,
    )

    set_value(
        ["Dependents"],
        1 if dependents == "Yes" else 0,
    )

    # Tenure and charges

    set_value(
        ["tenure", "Tenure"],
        tenure,
    )

    set_value(
        ["MonthlyCharges", "Monthly Charges"],
        monthly_charges,
    )

    set_value(
        ["TotalCharges", "Total Charges"],
        monthly_charges * tenure,
    )

    # Services

    set_value(
        ["PhoneService"],
        1 if phone == "Yes" else 0,
    )

    set_value(
        ["MultipleLines"],
        1 if multiple_lines == "Yes" else 0,
    )

    set_value(
        ["PaperlessBilling"],
        1 if paperless == "Yes" else 0,
    )

    set_value(
        ["OnlineSecurity"],
        1 if security == "Yes" else 0,
    )

    set_value(
        ["OnlineBackup"],
        1 if backup == "Yes" else 0,
    )

    set_value(
        ["DeviceProtection"],
        1 if device == "Yes" else 0,
    )

    set_value(
        ["TechSupport"],
        1 if support == "Yes" else 0,
    )

    set_value(
        ["StreamingTV"],
        1 if streaming_tv == "Yes" else 0,
    )

    set_value(
        ["StreamingMovies"],
        1 if streaming_movies == "Yes" else 0,
    )

    # Existing preprocessing encodes InternetService:
    # No = 0, DSL = 1, Fiber optic = 2.

    set_value(
        ["InternetService", "Internet Service"],
        {
            "No": 0,
            "DSL": 1,
            "Fiber optic": 2,
        }[internet],
    )

    # Existing preprocessing encodes Contract:
    # Month-to-month = 0, One year = 1, Two year = 2.

    set_value(
        ["Contract"],
        {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2,
        }[contract],
    )

    # Payment-method one-hot columns.

    for column in feature_columns:

        normalized = normalize_name(column)

        if normalized.startswith("paymentmethod"):

            row.loc[0, column] = 0

    payment_candidates = [
        f"PaymentMethod_{payment}",
        f"Payment Method_{payment}",
    ]

    for candidate in payment_candidates:

        column = match_feature(
            feature_columns,
            [candidate],
        )

        if column is not None:
            row.loc[0, column] = 1

    return row.reindex(
        columns=feature_columns,
        fill_value=0,
    )


# Compact prediction form

with st.form("prediction_form"):

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12,
        )

    with p2:

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes"],
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        security = st.selectbox(
            "Online Security",
            ["No", "Yes"],
        )

        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes"],
        )

    with p3:

        device = st.selectbox(
            "Device Protection",
            ["No", "Yes"],
        )

        support = st.selectbox(
            "Tech Support",
            ["No", "Yes"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes"],
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

    with p4:

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=5.0,
        )

        prediction_model = st.selectbox(
            "Prediction Model",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(
                model_metrics.iloc[0]["Model"]
            ),
        )

    predict_button = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True,
    )


if predict_button:

    customer_row = make_prediction_row(
        feature_columns=X.columns.tolist(),
        gender=gender,
        senior=senior,
        partner=partner,
        dependents=dependents,
        tenure=tenure,
        phone=phone,
        multiple_lines=multiple_lines,
        internet=internet,
        security=security,
        backup=backup,
        device=device,
        support=support,
        streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
        contract=contract,
        paperless=paperless,
        payment=payment,
        monthly_charges=monthly_charges,
    )

    selected_model = trained_models[
        prediction_model
    ]

    prediction = int(
        selected_model.predict(
            customer_row
        )[0]
    )

    probabilities = selected_model.predict_proba(
        customer_row
    )[0]

    churn_probability = float(
        probabilities[1]
    )

    retention_probability = float(
        probabilities[0]
    )

    st.markdown("#### Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Selected Model",
            prediction_model,
        )

    with r2:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}",
        )

    with r3:

        st.metric(
            "Retention Probability",
            f"{retention_probability:.2%}",
        )

    result_fig = go.Figure()

    result_fig.add_trace(
        go.Bar(
            x=["Retained", "Churned"],
            y=[
                retention_probability * 100,
                churn_probability * 100,
            ],
            text=[
                f"{retention_probability:.1%}",
                f"{churn_probability:.1%}",
            ],
            textposition="outside",
            marker_color=[
                "#4D9A91",
                "#D97878",
            ],
        )
    )

    result_fig.update_layout(
        title="Predicted Customer Outcome",
        yaxis_title="Probability (%)",
        yaxis=dict(
            range=[0, 110],
        ),
        showlegend=False,
    )

    st.plotly_chart(
        make_chart(
            result_fig,
            height=240,
        ),
        use_container_width=True,
    )

    if prediction == 1:

        st.error(
            "The selected model predicts that this customer "
            "may churn. Consider reviewing customer satisfaction, "
            "service issues, pricing, and contract options."
        )

    else:

        st.success(
            "The selected model predicts that this customer "
            "may be retained. Continue monitoring satisfaction "
            "and service usage."
        )

    st.caption(
        "This is a model-based estimate, not a guarantee. "
        "Check that the form's feature encoding matches "
        "the preprocessing used during training."
    )

    prediction_export = customer_row.copy()

    prediction_export["Selected_Model"] = prediction_model

    prediction_export["Predicted_Churn"] = prediction

    prediction_export["Churn_Probability"] = churn_probability

    st.download_button(
        "Download Prediction",
        data=prediction_export.to_csv(
            index=False,
        ).encode("utf-8"),
        file_name="customer_prediction.csv",
        mime="text/csv",
    )


# ============================================================
# 13. ANALYTICS CHARTS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">Customer Analytics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Explore churn patterns and customer behaviour.'
    '</div>',
    unsafe_allow_html=True,
)


# Chart 1: Churn distribution

chart1, chart2 = st.columns(2)

with chart1:

    st.markdown("#### Customer Retention Distribution")

    counts = (
        filtered_df[churn_col]
        .map({
            0: "Retained",
            1: "Churned",
        })
        .value_counts()
        .reindex(
            ["Retained", "Churned"],
            fill_value=0,
        )
        .reset_index()
    )

    counts.columns = [
        "Status",
        "Customers",
    ]

    fig = px.pie(
        counts,
        names="Status",
        values="Customers",
        hole=0.62,
        color="Status",
        color_discrete_map={
            "Retained": "#4D9A91",
            "Churned": "#D97878",
        },
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    st.plotly_chart(
        make_chart(fig),
        use_container_width=True,
    )


# Chart 2: Customer status

with chart2:

    st.markdown("#### Customer Status")

    fig = px.bar(
        counts,
        x="Status",
        y="Customers",
        color="Status",
        text="Customers",
        color_discrete_map={
            "Retained": "#4D9A91",
            "Churned": "#D97878",
        },
    )

    fig.update_traces(
        textposition="outside",
    )

    st.plotly_chart(
        make_chart(fig),
        use_container_width=True,
    )


# Chart 3: Contract

chart3, chart4 = st.columns(2)

with chart3:

    st.markdown("#### Churn by Contract")

    if contract_col:

        contract_summary = (
            filtered_df
            .groupby(
                [
                    contract_col,
                    churn_col,
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="Customers")
        )

        contract_summary["Status"] = (
            contract_summary[churn_col]
            .map({
                0: "Retained",
                1: "Churned",
            })
        )

        fig = px.bar(
            contract_summary,
            x=contract_col,
            y="Customers",
            color="Status",
            barmode="group",
            color_discrete_map={
                "Retained": "#4D9A91",
                "Churned": "#D97878",
            },
            labels={
                contract_col: "Contract",
            },
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True,
        )

    else:

        st.info("Contract column is unavailable.")


# Chart 4: Internet service

with chart4:

    st.markdown("#### Churn by Internet Service")

    if internet_col:

        internet_summary = (
            filtered_df
            .groupby(
                [
                    internet_col,
                    churn_col,
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="Customers")
        )

        internet_summary["Status"] = (
            internet_summary[churn_col]
            .map({
                0: "Retained",
                1: "Churned",
            })
        )

        fig = px.bar(
            internet_summary,
            x=internet_col,
            y="Customers",
            color="Status",
            barmode="group",
            color_discrete_map={
                "Retained": "#4D9A91",
                "Churned": "#D97878",
            },
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True,
        )

    else:

        st.info("Internet service column is unavailable.")


# Chart 5: Tenure

chart5, chart6 = st.columns(2)

with chart5:

    st.markdown("#### Customer Tenure Distribution")

    if tenure_col:

        tenure_data = filtered_df.copy()

        tenure_data["Status"] = (
            tenure_data[churn_col]
            .map({
                0: "Retained",
                1: "Churned",
            })
        )

        fig = px.histogram(
            tenure_data,
            x=tenure_col,
            color="Status",
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={
                "Retained": "#4D9A91",
                "Churned": "#D97878",
            },
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True,
        )

    else:

        st.info("Tenure column is unavailable.")


# Chart 6: Monthly charges

with chart6:

    st.markdown("#### Monthly Charges by Churn")

    if monthly_col:

        charges_data = filtered_df.copy()

        charges_data["Status"] = (
            charges_data[churn_col]
            .map({
                0: "Retained",
                1: "Churned",
            })
        )

        fig = px.box(
            charges_data,
            x="Status",
            y=monthly_col,
            color="Status",
            color_discrete_map={
                "Retained": "#4D9A91",
                "Churned": "#D97878",
            },
            points=False,
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True,
        )

    else:

        st.info("Monthly charges column is unavailable.")


# Chart 7: Total charges

if total_col:

    st.markdown("#### Total Charges Distribution")

    total_data = filtered_df.copy()

    total_data[total_col] = pd.to_numeric(
        total_data[total_col],
        errors="coerce",
    )

    total_data["Status"] = (
        total_data[churn_col]
        .map({
            0: "Retained",
            1: "Churned",
        })
    )

    total_data = total_data.dropna(
        subset=[total_col],
    )

    if not total_data.empty:

        fig = px.histogram(
            total_data,
            x=total_col,
            color="Status",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                "Retained": "#4D9A91",
                "Churned": "#D97878",
            },
        )

        st.plotly_chart(
            make_chart(fig),
            use_container_width=True,
        )


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">Model Performance Comparison</div>',
    unsafe_allow_html=True,
)

best_model = model_metrics.iloc[0]["Model"]

best_auc = model_metrics.iloc[0]["ROC-AUC"]

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Models Compared",
        len(model_metrics),
    )

with m2:

    st.metric(
        "Top Model by ROC-AUC",
        best_model,
    )

with m3:

    st.metric(
        "Top ROC-AUC",
        f"{best_auc:.2%}" if pd.notna(best_auc) else "N/A",
    )


# Model metrics table

st.markdown("#### Evaluation Metrics")

formatted_metrics = model_metrics.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
]:

    formatted_metrics[column] = formatted_metrics[
        column
    ].apply(
        lambda value: (
            f"{value:.2%}"
            if pd.notna(value)
            else "N/A"
        )
    )

st.dataframe(
    formatted_metrics,
    use_container_width=True,
    hide_index=True,
)


# Model comparison chart

st.markdown("#### Performance Metrics by Model")

long_metrics = model_metrics.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ],
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    long_metrics,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    range_y=[0, 1],
)

st.plotly_chart(
    make_chart(
        fig,
        height=330,
    ),
    use_container_width=True,
)


# ============================================================
# 15. ROC CURVES
# ============================================================

st.markdown("#### ROC Curve Comparison")

roc_fig = go.Figure()

for name in trained_models:

    probabilities = model_probabilities[name]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{name} (AUC {auc:.3f})",
        )
    )


roc_fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(
            dash="dash",
            color="#94A3B8",
        ),
    )
)

roc_fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

st.plotly_chart(
    make_chart(
        roc_fig,
        height=330,
    ),
    use_container_width=True,
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

st.markdown("#### Confusion Matrix")

confusion_model = st.selectbox(
    "Select model",
    list(trained_models.keys()),
    key="confusion_model",
)

cm = confusion_matrix(
    y_test,
    model_predictions[confusion_model],
    labels=[0, 1],
)

fig = px.imshow(
    cm,
    text_auto=True,
    x=[
        "Predicted Retained",
        "Predicted Churned",
    ],
    y=[
        "Actual Retained",
        "Actual Churned",
    ],
    color_continuous_scale=[
        "#EDF3F8",
        "#294C6B",
    ],
    aspect="auto",
    title=confusion_model,
)

st.plotly_chart(
    make_chart(
        fig,
        height=300,
    ),
    use_container_width=True,
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

st.markdown("#### Feature Importance")

importance_model_name = st.selectbox(
    "Choose a model",
    [
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
    ],
    key="importance_model",
)

importance_estimator = trained_models[
    importance_model_name
].named_steps["model"]

if hasattr(
    importance_estimator,
    "feature_importances_",
):

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importance_estimator.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    top_features = importance_df.head(12).sort_values(
        "Importance",
        ascending=True,
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            "#DCE8F2",
            "#294C6B",
        ],
        title=f"Most Important Features — {importance_model_name}",
    )

    st.plotly_chart(
        make_chart(
            fig,
            height=350,
        ),
        use_container_width=True,
    )

    st.download_button(
        "Download Feature Importance",
        data=importance_df.to_csv(
            index=False,
        ).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
    )


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

st.markdown("#### Feature Correlation")

numeric_data = filtered_df.select_dtypes(
    include=np.number,
).copy()

if numeric_data.shape[1] >= 2:

    correlation = numeric_data.corr(
        numeric_only=True,
    )

    if correlation.shape[0] > 12:

        if churn_col in correlation.columns:

            important = (
                correlation[churn_col]
                .drop(labels=[churn_col])
                .abs()
                .sort_values(ascending=False)
                .head(11)
                .index.tolist()
            )

            selected = important + [churn_col]

            correlation = correlation.loc[
                selected,
                selected,
            ]

        else:

            correlation = correlation.iloc[:12, :12]

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )

    st.plotly_chart(
        make_chart(
            fig,
            height=350,
        ),
        use_container_width=True,
    )
# ============================================================
# 19. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align: center;
        color: #627D98;
        font-size: 11px;
        padding: 8px;
    ">
        Customer Churn Intelligence |
        Python • Pandas • Scikit-learn • Plotly • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
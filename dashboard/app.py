# ============================================================
# CUSTOMER CHURN INTELLIGENCE DASHBOARD
# Power BI-Inspired Full-Width Streamlit Application
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. CHART COLORS ONLY
#    These colors are NOT applied to the dashboard UI.
# ============================================================

CHART_PINK = "#F4A6C1"
CHART_LIGHT_GREEN = "#A8D5B5"

CHART_PALETTE = [
    CHART_PINK,
    CHART_LIGHT_GREEN,
]

CHURN_COLORS = {
    "Churned": CHART_PINK,
    "Retained": CHART_LIGHT_GREEN,
}

px.defaults.color_discrete_sequence = CHART_PALETTE


# ============================================================
# 3. FULL-WIDTH LAYOUT
#    Keep the existing dashboard UI styling unchanged.
# ============================================================

st.markdown(
    """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }

        [data-testid="stMetric"] {
            padding: 0.25rem 0;
        }

        .section-heading {
            margin-top: 1.4rem;
            margin-bottom: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. HEADER
# ============================================================

st.title("Customer Churn Intelligence Dashboard")

st.caption(
    "AI-Powered Customer Insights | Predictive Analytics | "
    "Retention Intelligence"
)

st.markdown("---")


# ============================================================
# 5. ROBUST DATASET DISCOVERY
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

POSSIBLE_DATA_NAMES = [
    "cleaned_churn_data.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Telco-Customer-Churn.csv",
    "telco_customer_churn.csv",
    "customer_churn.csv",
    "churn.csv",
    "data.csv",
]

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "site-packages",
    "dist",
    "build",
}


def find_dataset():
    """
    Search likely project directories and their subdirectories
    for a supported customer churn CSV file.
    """

    search_roots = [
        APP_DIR,
        PROJECT_DIR,
        PROJECT_DIR.parent,
        PROJECT_DIR.parent.parent,
    ]

    checked_roots = set()

    # First search for expected filenames.
    for root in search_roots:
        if not root.exists():
            continue

        root = root.resolve()

        if root in checked_roots:
            continue

        checked_roots.add(root)

        for filename in POSSIBLE_DATA_NAMES:
            direct_path = root / filename

            if direct_path.is_file():
                return direct_path

        try:
            for path in root.rglob("*.csv"):
                if any(
                    part.lower() in IGNORED_DIRS
                    for part in path.parts
                ):
                    continue

                if path.name.lower() in {
                    name.lower()
                    for name in POSSIBLE_DATA_NAMES
                }:
                    return path

        except (PermissionError, OSError):
            continue

    # If the filename differs, search for another CSV that
    # appears to contain customer churn data.
    for root in checked_roots:
        try:
            for path in root.rglob("*.csv"):
                if any(
                    part.lower() in IGNORED_DIRS
                    for part in path.parts
                ):
                    continue

                try:
                    preview = pd.read_csv(
                        path,
                        nrows=5,
                        low_memory=False,
                    )

                    normalized_columns = {
                        str(column).strip().lower()
                        for column in preview.columns
                    }

                    has_churn = any(
                        column in normalized_columns
                        for column in [
                            "churn",
                            "exited",
                            "customerchurn",
                            "churnstatus",
                            "target",
                        ]
                    )

                    if has_churn:
                        return path

                except Exception:
                    continue

        except (PermissionError, OSError):
            continue

    return None


DATA_PATH = find_dataset()


# ============================================================
# 6. LOAD AND CLEAN DATA
# ============================================================

@st.cache_data(show_spinner="Loading customer churn dataset...")
def load_data(file_path, modified_time):
    """
    Load CSV and perform basic data cleaning.
    modified_time is included in the cache key so that
    changes to the CSV invalidate the cached result.
    """

    del modified_time

    data = pd.read_csv(
        file_path,
        low_memory=False,
    )

    # Clean column names.
    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    # Remove duplicate columns.
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # Remove completely empty rows and columns.
    data = data.dropna(axis=0, how="all")
    data = data.dropna(axis=1, how="all")

    # Normalize common whitespace-only values.
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].apply(
            lambda value: (
                np.nan
                if isinstance(value, str) and not value.strip()
                else value.strip()
                if isinstance(value, str)
                else value
            )
        )

    # Convert TotalCharges to numeric if available.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce",
        )

    return data


if DATA_PATH is None:
    st.error(
        "Customer churn dataset could not be found."
    )

    st.write(
        "Place your CSV file in the project folder or its "
        "data subfolder."
    )

    st.write("Supported filenames include:")

    for filename in POSSIBLE_DATA_NAMES:
        st.write(f"- `{filename}`")

    st.write("**Folders searched:**")

    for root in [
        APP_DIR,
        PROJECT_DIR,
        PROJECT_DIR.parent,
        PROJECT_DIR.parent.parent,
    ]:
        st.code(str(root))

    st.stop()


try:
    raw_data = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error("The dataset was found but could not be loaded.")
    st.exception(error)
    st.stop()


if raw_data.empty:
    st.error("The dataset contains no records.")
    st.stop()


# ============================================================
# 7. DETECT TARGET COLUMN
# ============================================================

TARGET_CANDIDATES = [
    "Churn",
    "Exited",
    "CustomerChurn",
    "ChurnStatus",
    "Target",
]

target_lookup = {
    str(column).strip().lower(): column
    for column in raw_data.columns
}

TARGET_COLUMN = None

for candidate in TARGET_CANDIDATES:
    if candidate.lower() in target_lookup:
        TARGET_COLUMN = target_lookup[candidate.lower()]
        break


if TARGET_COLUMN is None:
    st.error(
        "Could not identify the churn target column."
    )

    st.write(
        "The dataset must contain a target column such as "
        "`Churn`, `Exited`, `CustomerChurn`, `ChurnStatus`, "
        "or `Target`."
    )

    st.stop()


# ============================================================
# 8. NORMALIZE CHURN LABELS
# ============================================================

def normalize_churn_target(series):
    """
    Convert common churn labels to binary values:
    1 = Churned
    0 = Retained
    """

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_missing = numeric_values.dropna()

    if (
        len(non_missing) > 0
        and set(non_missing.unique()).issubset({0, 1})
    ):
        return numeric_values.astype("Int64")

    text_values = series.astype("string").str.strip().str.lower()

    positive_labels = {
        "yes",
        "true",
        "1",
        "churn",
        "churned",
        "exited",
        "left",
        "positive",
    }

    negative_labels = {
        "no",
        "false",
        "0",
        "retained",
        "stay",
        "stayed",
        "not churned",
        "negative",
    }

    normalized = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    normalized[text_values.isin(positive_labels)] = 1
    normalized[text_values.isin(negative_labels)] = 0

    # Handle two-class datasets with unexpected label names.
    unknown_mask = (
        text_values.notna()
        & normalized.isna()
    )

    unknown_values = text_values[unknown_mask].unique()

    if len(unknown_values) == 2:
        # Treat the first class as negative and the second
        # as positive in sorted order.
        sorted_values = sorted(
            [str(value) for value in unknown_values]
        )

        normalized[text_values == sorted_values[0]] = 0
        normalized[text_values == sorted_values[1]] = 1

    return normalized.astype("Int64")


clean_data = raw_data.copy()

clean_data[TARGET_COLUMN] = normalize_churn_target(
    clean_data[TARGET_COLUMN]
)

clean_data = clean_data.dropna(
    subset=[TARGET_COLUMN]
).copy()

clean_data[TARGET_COLUMN] = clean_data[
    TARGET_COLUMN
].astype(int)

if clean_data.empty:
    st.error("No usable target values were found.")
    st.stop()

if clean_data[TARGET_COLUMN].nunique() < 2:
    st.error(
        "The target column must contain both churn and "
        "non-churn examples to train classification models."
    )
    st.stop()


# ============================================================
# 9. IDENTIFY COMMON CUSTOMER FEATURES
# ============================================================

def find_column(data, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


CUSTOMER_ID_COLUMN = find_column(
    clean_data,
    [
        "customerID",
        "CustomerID",
        "customer_id",
        "Customer Id",
        "ID",
    ],
)

CONTRACT_COLUMN = find_column(
    clean_data,
    [
        "Contract",
        "ContractType",
        "Contract Type",
    ],
)

INTERNET_COLUMN = find_column(
    clean_data,
    [
        "InternetService",
        "Internet Service",
        "Internet_Service",
    ],
)

TENURE_COLUMN = find_column(
    clean_data,
    [
        "tenure",
        "Tenure",
        "CustomerTenure",
    ],
)

MONTHLY_CHARGES_COLUMN = find_column(
    clean_data,
    [
        "MonthlyCharges",
        "Monthly Charges",
        "Monthly_Charges",
    ],
)

TOTAL_CHARGES_COLUMN = find_column(
    clean_data,
    [
        "TotalCharges",
        "Total Charges",
        "Total_Charges",
    ],
)


# ============================================================
# 10. DASHBOARD FILTERS
# ============================================================

st.subheader("Dashboard Filters")

# Three equal-width filters; Customer ID search is removed.
filter_col1, filter_col2, filter_col3 = st.columns(3)

filtered_data = clean_data.copy()


# --------------------- Contract Filter ----------------------

with filter_col1:
    if CONTRACT_COLUMN:
        contract_options = sorted(
            filtered_data[CONTRACT_COLUMN]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_contracts = st.multiselect(
            "Contract Type",
            options=contract_options,
            default=contract_options,
        )

        filtered_data = filtered_data[
            filtered_data[CONTRACT_COLUMN]
            .astype(str)
            .isin(selected_contracts)
        ]

    else:
        st.caption("Contract type unavailable.")


# --------------------- Internet Service Filter --------------

with filter_col2:
    if INTERNET_COLUMN:
        internet_options = sorted(
            filtered_data[INTERNET_COLUMN]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_internet = st.multiselect(
            "Internet Service",
            options=internet_options,
            default=internet_options,
        )

        filtered_data = filtered_data[
            filtered_data[INTERNET_COLUMN]
            .astype(str)
            .isin(selected_internet)
        ]

    else:
        st.caption("Internet service unavailable.")


# --------------------- Churn Status Filter ------------------

with filter_col3:
    churn_filter = st.selectbox(
        "Churn Status",
        options=[
            "All Customers",
            "Churned",
            "Retained",
        ],
    )

    if churn_filter == "Churned":
        filtered_data = filtered_data[
            filtered_data[TARGET_COLUMN] == 1
        ]

    elif churn_filter == "Retained":
        filtered_data = filtered_data[
            filtered_data[TARGET_COLUMN] == 0
        ]


# --------------------- Empty Filter Validation ---------------

if filtered_data.empty:
    st.warning(
        "No customers match the selected filters. "
        "Change the filters to see the dashboard."
    )
    st.stop()

# ============================================================
# 11. KPI CARDS
# ============================================================

st.markdown("---")
st.subheader("Customer Overview")

total_customers = len(filtered_data)

churned_customers = int(
    (filtered_data[TARGET_COLUMN] == 1).sum()
)

retained_customers = int(
    (filtered_data[TARGET_COLUMN] == 0).sum()
)

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers
    else 0
)

retention_rate = (
    retained_customers / total_customers * 100
    if total_customers
    else 0
)

if MONTHLY_CHARGES_COLUMN:
    average_monthly_charges = pd.to_numeric(
        filtered_data[MONTHLY_CHARGES_COLUMN],
        errors="coerce",
    ).mean()
else:
    average_monthly_charges = np.nan


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Total Customers",
    f"{total_customers:,}",
)

kpi2.metric(
    "Churned Customers",
    f"{churned_customers:,}",
)

kpi3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%",
)

kpi4.metric(
    "Retention Rate",
    f"{retention_rate:.2f}%",
)

kpi5.metric(
    "Avg. Monthly Charges",
    (
        f"${average_monthly_charges:,.2f}"
        if pd.notna(average_monthly_charges)
        else "N/A"
    ),
)


# ============================================================
# 12. CHURN ANALYTICS
# ============================================================

st.markdown("---")
st.subheader("Churn Analytics")

analytics_col1, analytics_col2, analytics_col3 = st.columns(3)


# --------------------- Churn Distribution -------------------

with analytics_col1:
    st.markdown("**Churn Distribution**")

    churn_counts = (
        filtered_data[TARGET_COLUMN]
        .map({
            1: "Churned",
            0: "Retained",
        })
        .value_counts()
        .reindex(["Churned", "Retained"], fill_value=0)
        .reset_index()
    )

    churn_counts.columns = [
        "Churn Status",
        "Customers",
    ]

    fig_churn = px.pie(
        churn_counts,
        names="Churn Status",
        values="Customers",
        hole=0.45,
        color="Churn Status",
        color_discrete_map=CHURN_COLORS,
    )

    fig_churn.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig_churn.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
    )

    st.plotly_chart(
        fig_churn,
        use_container_width=True,
    )


# --------------------- Contract Distribution ----------------

with analytics_col2:
    st.markdown("**Customers by Contract**")

    if CONTRACT_COLUMN:
        contract_counts = (
            filtered_data[CONTRACT_COLUMN]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Contract")
            .reset_index(name="Customers")
        )

        fig_contract = px.bar(
            contract_counts,
            x="Contract",
            y="Customers",
            color="Contract",
            color_discrete_sequence=CHART_PALETTE,
            text="Customers",
        )

        fig_contract.update_traces(
            textposition="outside",
        )

        fig_contract.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Contract Type",
            yaxis_title="Number of Customers",
        )

        st.plotly_chart(
            fig_contract,
            use_container_width=True,
        )

    else:
        st.info("Contract data is unavailable.")


# --------------------- Internet Service ---------------------

with analytics_col3:
    st.markdown("**Churn by Internet Service**")

    if INTERNET_COLUMN:
        internet_churn = filtered_data.copy()

        internet_churn["Churn Status"] = (
            internet_churn[TARGET_COLUMN]
            .map({
                1: "Churned",
                0: "Retained",
            })
        )

        internet_counts = (
            internet_churn.groupby(
                [INTERNET_COLUMN, "Churn Status"],
                dropna=False,
            )
            .size()
            .reset_index(name="Customers")
        )

        internet_counts[INTERNET_COLUMN] = (
            internet_counts[INTERNET_COLUMN]
            .fillna("Unknown")
            .astype(str)
        )

        fig_internet = px.bar(
            internet_counts,
            x=INTERNET_COLUMN,
            y="Customers",
            color="Churn Status",
            barmode="group",
            color_discrete_map=CHURN_COLORS,
        )

        fig_internet.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Internet Service",
            yaxis_title="Number of Customers",
            legend_title_text="",
        )

        st.plotly_chart(
            fig_internet,
            use_container_width=True,
        )

    else:
        st.info("Internet service data is unavailable.")


# ============================================================
# 13. CUSTOMER BEHAVIOUR
# ============================================================

st.markdown("---")
st.subheader("Customer Behaviour")

behaviour_col1, behaviour_col2 = st.columns(2)


# --------------------- Tenure Distribution ------------------

with behaviour_col1:
    st.markdown("**Customer Tenure Distribution**")

    if TENURE_COLUMN:
        tenure_data = filtered_data.copy()

        tenure_data[TENURE_COLUMN] = pd.to_numeric(
            tenure_data[TENURE_COLUMN],
            errors="coerce",
        )

        tenure_data = tenure_data.dropna(
            subset=[TENURE_COLUMN]
        )

        tenure_data["Churn Status"] = (
            tenure_data[TARGET_COLUMN]
            .map({
                1: "Churned",
                0: "Retained",
            })
        )

        fig_tenure = px.histogram(
            tenure_data,
            x=TENURE_COLUMN,
            color="Churn Status",
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map=CHURN_COLORS,
        )

        fig_tenure.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Tenure",
            yaxis_title="Number of Customers",
            legend_title_text="",
        )

        st.plotly_chart(
            fig_tenure,
            use_container_width=True,
        )

    else:
        st.info("Tenure data is unavailable.")


# --------------------- Monthly Charges ----------------------

with behaviour_col2:
    st.markdown("**Monthly Charges by Churn Status**")

    if MONTHLY_CHARGES_COLUMN:
        charges_data = filtered_data.copy()

        charges_data[MONTHLY_CHARGES_COLUMN] = pd.to_numeric(
            charges_data[MONTHLY_CHARGES_COLUMN],
            errors="coerce",
        )

        charges_data["Churn Status"] = (
            charges_data[TARGET_COLUMN]
            .map({
                1: "Churned",
                0: "Retained",
            })
        )

        charges_data = charges_data.dropna(
            subset=[MONTHLY_CHARGES_COLUMN]
        )

        fig_charges = px.box(
            charges_data,
            x="Churn Status",
            y=MONTHLY_CHARGES_COLUMN,
            color="Churn Status",
            color_discrete_map=CHURN_COLORS,
            points="outliers",
        )

        fig_charges.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Churn Status",
            yaxis_title="Monthly Charges",
            legend_title_text="",
        )

        fig_charges.update_layout(
            showlegend=False,
        )

        st.plotly_chart(
            fig_charges,
            use_container_width=True,
        )

    else:
        st.info("Monthly charges data is unavailable.")


# ============================================================
# 14. CUSTOMER SEGMENTATION
# ============================================================

st.markdown("---")
st.subheader("Customer Segmentation")

preferred_features = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
    "PaperlessBilling",
    "Contract",
    "InternetService",
    "gender",
    "Payment Method",
]

segmentation_features = []

for feature in preferred_features:
    actual_column = find_column(
        filtered_data,
        [feature],
    )

    if actual_column and actual_column != TARGET_COLUMN:
        if actual_column not in segmentation_features:
            segmentation_features.append(actual_column)

# Include other categorical columns if fewer than four
# useful segmentation features were found.
for column in filtered_data.columns:
    if column == TARGET_COLUMN:
        continue

    if column == CUSTOMER_ID_COLUMN:
        continue

    if column in segmentation_features:
        continue

    if (
        filtered_data[column].dtype == "object"
        or str(filtered_data[column].dtype) == "category"
        or filtered_data[column].nunique(dropna=True) <= 10
    ):
        if filtered_data[column].nunique(dropna=True) <= 10:
            segmentation_features.append(column)

    if len(segmentation_features) >= 4:
        break

segmentation_features = segmentation_features[:4]

if segmentation_features:
    segmentation_col1, segmentation_col2 = st.columns(2)

    for index, feature in enumerate(segmentation_features):
        target_col = (
            segmentation_col1
            if index % 2 == 0
            else segmentation_col2
        )

        with target_col:
            st.markdown(f"**Churn by {feature}**")

            segment_data = filtered_data.copy()

            segment_data[feature] = (
                segment_data[feature]
                .fillna("Unknown")
                .astype(str)
            )

            segment_data["Churn Status"] = (
                segment_data[TARGET_COLUMN]
                .map({
                    1: "Churned",
                    0: "Retained",
                })
            )

            segment_counts = (
                segment_data.groupby(
                    [feature, "Churn Status"],
                    dropna=False,
                )
                .size()
                .reset_index(name="Customers")
            )

            fig_segment = px.bar(
                segment_counts,
                x=feature,
                y="Customers",
                color="Churn Status",
                barmode="group",
                color_discrete_map=CHURN_COLORS,
            )

            fig_segment.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=feature,
                yaxis_title="Number of Customers",
                legend_title_text="",
            )

            st.plotly_chart(
                fig_segment,
                use_container_width=True,
            )

else:
    st.info(
        "No suitable categorical columns were found "
        "for customer segmentation."
    )


# ============================================================
# 15. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

st.markdown("---")
st.subheader("Machine Learning Model Performance")

model_data = clean_data.copy()

# Exclude the customer identifier and target from features.
excluded_columns = [TARGET_COLUMN]

if CUSTOMER_ID_COLUMN:
    excluded_columns.append(CUSTOMER_ID_COLUMN)

X = model_data.drop(
    columns=excluded_columns,
    errors="ignore",
).copy()

y = model_data[TARGET_COLUMN].astype(int).copy()

# Remove completely empty feature columns.
X = X.dropna(
    axis=1,
    how="all",
)

# Convert numeric-looking text columns into numeric columns.
for column in X.columns:
    if X[column].dtype == "object":
        converted = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        original_non_missing = X[column].notna().sum()
        converted_non_missing = converted.notna().sum()

        if (
            original_non_missing > 0
            and converted_non_missing / original_non_missing >= 0.90
        ):
            X[column] = converted

# Remove rows where the target or features are not usable.
valid_target_mask = y.notna()

X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()

if len(X) < 10:
    st.error(
        "There are too few usable records to train and "
        "evaluate the classification models."
    )
    st.stop()

if y.nunique() < 2:
    st.error(
        "The target must contain both churn classes."
    )
    st.stop()

# Remove constant columns.
constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

if constant_columns:
    X = X.drop(
        columns=constant_columns,
    )

if X.shape[1] == 0:
    st.error(
        "No usable feature columns remain after cleaning."
    )
    st.stop()

numeric_features = X.select_dtypes(
    include=["number", "bool"],
).columns.tolist()

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]

# Make categorical data consistent for encoding.
for column in categorical_features:
    X[column] = X[column].astype("object")


# ============================================================
# 16. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
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
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

transformers = []

if numeric_features:
    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        )
    )

if categorical_features:
    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


# ============================================================
# 17. TRAIN AND COMPARE CLASSIFICATION MODELS
# ============================================================

test_fraction = 0.20

class_counts = y.value_counts()

can_stratify = (
    y.nunique() == 2
    and class_counts.min() >= 2
)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=y if can_stratify else None,
    )

except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=None,
    )


if y_train.nunique() < 2 or y_test.nunique() < 2:
    st.error(
        "The training or testing split contains only one class. "
        "More examples of each churn class are required."
    )
    st.stop()


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    model_data_signature,
):
    """
    Fit the preprocessing pipeline and each model.
    Store metrics, predictions, probabilities, and fitted pipelines.
    """

    del model_data_signature

    fitted_models = {}
    metrics_records = {}
    predictions = {}
    probabilities = {}
    errors = {}

    local_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    for model_name, classifier in local_models.items():
        try:
            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "classifier",
                        classifier,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]

            elif hasattr(pipeline, "decision_function"):
                decision_scores = pipeline.decision_function(X_test)

                y_prob = (
                    decision_scores - decision_scores.min()
                ) / (
                    decision_scores.max()
                    - decision_scores.min()
                    + 1e-9
                )

            else:
                y_prob = y_pred.astype(float)

            metrics = {
                "Accuracy": accuracy_score(
                    y_test,
                    y_pred,
                ),

                "Precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),

                "Recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),

                "F1 Score": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            }

            try:
                metrics["ROC-AUC"] = roc_auc_score(
                    y_test,
                    y_prob,
                )

            except ValueError:
                metrics["ROC-AUC"] = np.nan

            fitted_models[model_name] = pipeline
            metrics_records[model_name] = metrics
            predictions[model_name] = y_pred
            probabilities[model_name] = y_prob

        except Exception as error:
            errors[model_name] = str(error)

    return (
        fitted_models,
        metrics_records,
        predictions,
        probabilities,
        errors,
    )


model_signature = (
    len(X),
    len(X.columns),
    tuple(X.columns),
    tuple(y.value_counts().sort_index().items()),
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime,
)

(
    fitted_models,
    model_metrics,
    model_predictions,
    model_probabilities,
    model_errors,
) = train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    model_signature,
)


if not model_metrics:
    st.error(
        "None of the models could be trained successfully."
    )

    for model_name, error_message in model_errors.items():
        st.write(f"**{model_name}:** {error_message}")

    st.stop()


# ============================================================
# 18. MODEL METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame.from_dict(
    model_metrics,
    orient="index",
)

metrics_df.index.name = "Model"

metrics_df = metrics_df.reset_index()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
]

for column in metric_columns:
    if column in metrics_df.columns:
        metrics_df[column] = metrics_df[column] * 100


metrics_df = metrics_df.sort_values(
    by="ROC-AUC",
    ascending=False,
    na_position="last",
).reset_index(drop=True)


# Choose the model with the highest available ROC-AUC.
if metrics_df["ROC-AUC"].notna().any():
    best_model_name = metrics_df.iloc[0]["Model"]

else:
    best_model_name = (
        metrics_df
        .sort_values(
            by="F1 Score",
            ascending=False,
        )
        .iloc[0]["Model"]
    )


best_model = fitted_models[best_model_name]

st.markdown("#### Model Performance Summary")

st.caption(
    "Models are evaluated on a held-out test set. "
    "Reported metrics are percentages."
)

display_metrics_df = metrics_df.copy()

for column in metric_columns:
    if column in display_metrics_df.columns:
        display_metrics_df[column] = display_metrics_df[
            column
        ].map(
            lambda value: (
                f"{value:.2f}%"
                if pd.notna(value)
                else "N/A"
            )
        )

st.dataframe(
    display_metrics_df,
    use_container_width=True,
    hide_index=True,
)

best_col1, best_col2 = st.columns(2)

with best_col1:
    st.metric(
        "Best Model",
        best_model_name,
    )

with best_col2:
    best_auc = model_metrics[best_model_name].get(
        "ROC-AUC",
        np.nan,
    )

    st.metric(
        "Best Model ROC-AUC",
        (
            f"{best_auc * 100:.2f}%"
            if pd.notna(best_auc)
            else "N/A"
        ),
    )


# ============================================================
# 19. MODEL COMPARISON CHARTS
# ============================================================

st.markdown("---")
st.subheader("Model Comparison")

comparison_col1, comparison_col2 = st.columns(2)


with comparison_col1:
    st.markdown("**Classification Metrics Comparison**")

    comparison_long = metrics_df.melt(
        id_vars=["Model"],
        value_vars=[
            column
            for column in metric_columns
            if column != "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Score",
    )

    fig_model_metrics = px.bar(
        comparison_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=CHART_PALETTE,
        hover_data={
            "Score": ":.2f",
        },
    )

    fig_model_metrics.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Model",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        legend_title_text="",
    )

    st.plotly_chart(
        fig_model_metrics,
        use_container_width=True,
    )


with comparison_col2:
    st.markdown("**ROC-AUC Comparison**")

    auc_data = metrics_df[
        ["Model", "ROC-AUC"]
    ].dropna()

    fig_auc = px.bar(
        auc_data,
        x="Model",
        y="ROC-AUC",
        color="Model",
        color_discrete_sequence=CHART_PALETTE,
        text="ROC-AUC",
    )

    fig_auc.update_traces(
        texttemplate="%{y:.2f}%",
        textposition="outside",
    )

    fig_auc.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Model",
        yaxis_title="ROC-AUC (%)",
        yaxis_range=[
            0,
            min(
                110,
                max(
                    100,
                    (
                        float(auc_data["ROC-AUC"].max()) + 10
                        if not auc_data.empty
                        else 100
                    ),
                ),
            ),
        ],
        showlegend=False,
    )

    st.plotly_chart(
        fig_auc,
        use_container_width=True,
    )


# ============================================================
# 20. ROC CURVES
# ============================================================

st.markdown("---")
st.subheader("ROC Curve Analysis")

fig_roc = go.Figure()

for index, model_name in enumerate(fitted_models):
    if model_name not in model_probabilities:
        continue

    probabilities_for_model = model_probabilities[model_name]

    try:
        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities_for_model,
        )

        auc_value = roc_auc_score(
            y_test,
            probabilities_for_model,
        )

        line_color = CHART_PALETTE[
            index % len(CHART_PALETTE)
        ]

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc_value:.3f})",
                line=dict(
                    color=line_color,
                    width=2,
                ),
            )
        )

    except Exception:
        continue


fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(
            color="#808080",
            width=1,
            dash="dash",
        ),
    )
)

fig_roc.update_layout(
    height=480,
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis_range=[0, 1],
    yaxis_range=[0, 1.02],
    margin=dict(l=10, r=10, t=20, b=10),
    legend_title_text="Models",
)

st.plotly_chart(
    fig_roc,
    use_container_width=True,
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

st.markdown("---")
st.subheader("Confusion Matrix")

selected_confusion_model = st.selectbox(
    "Select Model for Confusion Matrix",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(
        best_model_name
    ),
)

selected_predictions = model_predictions[
    selected_confusion_model
]

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1],
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Retained",
        "Actual Churned",
    ],
    columns=[
        "Predicted Retained",
        "Predicted Churned",
    ],
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=[
        CHART_LIGHT_GREEN,
        CHART_PINK,
    ],
)

fig_cm.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
    coloraxis_colorbar_title="Customers",
)

st.plotly_chart(
    fig_cm,
    use_container_width=True,
)


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

st.markdown("---")
st.subheader("Feature Importance")

classifier = best_model.named_steps["classifier"]
fitted_preprocessor = best_model.named_steps["preprocessor"]

importance_values = None

if hasattr(classifier, "feature_importances_"):
    importance_values = classifier.feature_importances_

elif hasattr(classifier, "coef_"):
    importance_values = np.abs(
        classifier.coef_[0]
    )


if importance_values is not None:
    try:
        feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )

        feature_names = [
            name.replace("numeric__", "")
            .replace("categorical__", "")
            for name in feature_names
        ]

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values,
            }
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False,
        ).head(15)

        fig_importance = px.bar(
            importance_df.sort_values(
                by="Importance",
                ascending=True,
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[
                CHART_LIGHT_GREEN,
                CHART_PINK,
            ],
        )

        fig_importance.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Importance",
            yaxis_title="Feature",
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
        )

    except Exception as error:
        st.info(
            "Feature names could not be extracted for "
            "this model."
        )

else:
    st.info(
        "The selected model does not provide native "
        "feature importance or coefficients."
    )


# ============================================================
# 23. CORRELATION HEATMAP
# ============================================================

st.markdown("---")
st.subheader("Feature Correlation Heatmap")

numeric_correlation_data = clean_data.select_dtypes(
    include=["number"],
).copy()

if TARGET_COLUMN not in numeric_correlation_data.columns:
    numeric_correlation_data[TARGET_COLUMN] = clean_data[
        TARGET_COLUMN
    ]

# Remove the customer identifier from the correlation plot.
if CUSTOMER_ID_COLUMN in numeric_correlation_data.columns:
    numeric_correlation_data = numeric_correlation_data.drop(
        columns=[CUSTOMER_ID_COLUMN],
        errors="ignore",
    )

# Limit heatmap columns for readability.
if numeric_correlation_data.shape[1] > 20:
    correlation_strength = (
        numeric_correlation_data.corr(numeric_only=True)[
            TARGET_COLUMN
        ]
        .abs()
        .sort_values(ascending=False)
    )

    selected_correlation_columns = (
        correlation_strength.head(20).index.tolist()
    )

    numeric_correlation_data = (
        numeric_correlation_data[
            selected_correlation_columns
        ]
    )

if numeric_correlation_data.shape[1] >= 2:
    correlation_matrix = numeric_correlation_data.corr(
        numeric_only=True
    )

    fig_correlation = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale=[
            CHART_LIGHT_GREEN,
            "#FFFFFF",
            CHART_PINK,
        ],
    )

    fig_correlation.update_layout(
        height=max(
            450,
            min(
                750,
                35 * len(correlation_matrix.columns) + 200,
            ),
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar_title="Correlation",
    )

    st.plotly_chart(
        fig_correlation,
        use_container_width=True,
    )

else:
    st.info(
        "At least two numeric columns are required "
        "to create the correlation heatmap."
    )


# ============================================================
# 24. CUSTOMER CHURN PREDICTION
# ============================================================

st.markdown("---")
st.subheader("Predict Customer Churn")

st.write(
    "Enter customer details below to estimate the likelihood "
    "of churn using the selected machine learning model."
)

prediction_model_name = st.selectbox(
    "Prediction Model",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(
        best_model_name
    ),
)

prediction_model = fitted_models[
    prediction_model_name
]

with st.form("customer_prediction_form"):
    st.markdown("#### Customer Information")

    prediction_inputs = {}

    input_columns = st.columns(3)

    for index, column in enumerate(X.columns):
        with input_columns[index % 3]:
            original_series = X[column]

            if column in numeric_features:
                numeric_series = pd.to_numeric(
                    original_series,
                    errors="coerce",
                )

                median_value = numeric_series.median()

                if pd.isna(median_value):
                    median_value = 0.0

                minimum_value = numeric_series.min()
                maximum_value = numeric_series.max()

                if pd.isna(minimum_value):
                    minimum_value = 0.0

                if pd.isna(maximum_value):
                    maximum_value = 0.0

                if minimum_value > maximum_value:
                    minimum_value, maximum_value = (
                        maximum_value,
                        minimum_value,
                    )

                if minimum_value == maximum_value:
                    prediction_inputs[column] = st.number_input(
                        column,
                        value=float(median_value),
                        key=f"prediction_{column}",
                    )

                else:
                    prediction_inputs[column] = st.number_input(
                        column,
                        min_value=float(minimum_value),
                        max_value=float(maximum_value),
                        value=float(
                            min(
                                max(
                                    median_value,
                                    minimum_value,
                                ),
                                maximum_value,
                            )
                        ),
                        key=f"prediction_{column}",
                    )

            else:
                category_options = (
                    original_series.dropna()
                    .astype(str)
                    .value_counts()
                    .index.tolist()
                )

                if not category_options:
                    category_options = ["Unknown"]

                prediction_inputs[column] = st.selectbox(
                    column,
                    options=category_options,
                    key=f"prediction_{column}",
                )

    predict_button = st.form_submit_button(
        "Predict Churn",
        use_container_width=True,
    )


if predict_button:
    try:
        prediction_df = pd.DataFrame(
            [prediction_inputs],
            columns=X.columns,
        )

        for column in numeric_features:
            prediction_df[column] = pd.to_numeric(
                prediction_df[column],
                errors="coerce",
            )

        for column in categorical_features:
            prediction_df[column] = (
                prediction_df[column].astype("object")
            )

        predicted_class = int(
            prediction_model.predict(
                prediction_df
            )[0]
        )

        if hasattr(prediction_model, "predict_proba"):
            predicted_probability = float(
                prediction_model.predict_proba(
                    prediction_df
                )[0][1]
            )

        else:
            predicted_probability = float(
                predicted_class
            )

        prediction_result_col1, prediction_result_col2 = (
            st.columns(2)
        )

        with prediction_result_col1:
            st.metric(
                "Predicted Churn Probability",
                f"{predicted_probability * 100:.2f}%",
            )

        with prediction_result_col2:
            st.metric(
                "Prediction",
                (
                    "Churned"
                    if predicted_class == 1
                    else "Retained"
                ),
            )

        if predicted_class == 1:
            st.warning(
                "The model predicts that this customer "
                "may churn. Consider reviewing the customer's "
                "service experience, contract, and billing."
            )

        else:
            st.success(
                "The model predicts that this customer "
                "is likely to be retained."
            )

        st.caption(
            "This is a machine learning estimate, not a "
            "guarantee of future customer behaviour."
        )

    except Exception as error:
        st.error(
            "Prediction could not be completed. "
            "Check the input values and try again."
        )

        st.exception(error)


# ============================================================
# 25. EXPORT MODEL RESULTS
#     Retained. This exports model metrics only.
#     Dataset Explorer and customer-data downloads are removed.
# ============================================================

st.markdown("---")
st.subheader("Export Model Results")

st.write(
    "Download the model comparison metrics for your report "
    "or further analysis."
)

export_metrics_df = pd.DataFrame.from_dict(
    model_metrics,
    orient="index",
)

export_metrics_df.index.name = "Model"

export_metrics_df = export_metrics_df.reset_index()

for column in metric_columns:
    if column in export_metrics_df.columns:
        export_metrics_df[column] = (
            export_metrics_df[column] * 100
        ).round(2)

export_metrics_csv = export_metrics_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Model Comparison Results (CSV)",
    data=export_metrics_csv,
    file_name="customer_churn_model_results.csv",
    mime="text/csv",
    use_container_width=True,
)
# ============================================================
# 26. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Intelligence Dashboard | "
    "Machine Learning | Predictive Analytics"
)
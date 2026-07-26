# 📊 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts customer churn for a telecom company using the Telco Customer Churn dataset. The project includes data preprocessing, model training, customer churn prediction, visualizations, and an interactive Streamlit dashboard.

## 🎯 Features

- 🔮 Predict customer churn using a trained Random Forest model
- 📊 Interactive Streamlit dashboard for prediction and analytics
- 📈 Data preprocessing and feature engineering pipeline
- 📉 Visualizations for churn distribution, contract type, tenure, and feature importance
- 🏆 Model performance with Accuracy, Precision, Recall, F1-Score, and ROC-AUC

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib

## 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **77.22%** |
| Precision | **55.70%** |
| Recall | **69.25%** |
| F1-Score | **61.74%** |
| ROC-AUC | **83.24%** |

## 📁 Project Structure

```
Customer-churn-prediction/
│
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── trained/
├── reports/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── visualization.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Clone the repository

```bash
git clone https://github.com/TishaSutariya/Customer-churn-prediction-.git
cd Customer-churn-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

## 📌 Dataset

- IBM Telco Customer Churn Dataset
- 7,043 customer records
- 21 input features
- Binary churn prediction (Yes/No)

## 📷 Dashboard

Add screenshots of your dashboard here.

## 📜 License

This project is created for learning and educational purposes.
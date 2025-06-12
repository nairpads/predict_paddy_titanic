import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction + EDA")

train_file = st.file_uploader("Upload train.csv", type=["csv"])
test_file = st.file_uploader("Upload test.csv", type=["csv"])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # --- Preprocessing ---
    for dataset in [train, test]:
        dataset.columns = dataset.columns.str.strip().str.lower()
        if 'age' in dataset.columns:
            dataset['age'] = dataset['age'].fillna(dataset['age'].median())
        if 'embarked' in dataset.columns:
            mode = dataset['embarked'].mode()
            dataset['embarked'] = dataset['embarked'].fillna(mode[0] if not mode.empty else 's')
            dataset['embarked'] = dataset['embarked'].astype(str).str.lower().map({'s': 0, 'c': 1, 'q': 2})
            dataset['embarked'] = dataset['embarked'].fillna(-1).astype(int)
        if 'sex' in dataset.columns:
            dataset['sex'] = dataset['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
            dataset['sex'] = dataset['sex'].fillna(-1).astype(int)
    if 'fare' in test.columns:
        test['fare'] = test['fare'].fillna(test['fare'].median())

    # Keep passenger id for output
    test_passenger_id = test['passengerid']

    drop_cols = [c for c in ['name', 'ticket', 'cabin', 'passengerid'] if c in train.columns]
    train = train.drop(drop_cols, axis=1)
    test = test.drop([c for c in drop_cols if c in test.columns], axis=1)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(train.head())

    # --- EDA ---
    st.subheader("📊 Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='survived', data=train, ax=ax1)
    st.pyplot(fig1)

    st.subheader("👩‍🦰🧔 Survival by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='sex', hue='survived', data=train, ax=ax2)
    st.pyplot(fig2)

    st.subheader("🛌 Survival by Passenger Class")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='pclass', hue='survived', data=train, ax=ax3)
    st.pyplot(fig3)

    st.subheader("🎂 Age Distribution with Survival")
    if train['age'].dropna().shape[0] > 1:
        fig4, ax4 = plt.subplots()
        sns.histplot(data=train, x='age', hue='survived', bins=30, kde=True, ax=ax4)
        st.pyplot(fig4)
    else:
        st.write("Not enough non-null 'age' values to plot distribution with KDE.")

    # --- Data Split ---
    X = train.drop("survived", axis=1)
    y = train["survived"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Strict Data Cleaning ---
    # Only keep numeric columns in X
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    test = test.select_dtypes(include=[np.number])

    # Replace inf/nan and drop any row with nan in X or y
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_val = pd.to_numeric(y_val, errors='coerce')

    # Drop rows with nan in X_train or y_train
    mask = X_train.notnull().all(axis=1) & y_train.notnull()
    X_train = X_train[mask]
    y_train = y_train[mask].astype(int)

    mask_val = X_val.notnull().all(axis=1) & y_val.notnull()
    X_val = X_val[mask_val]
    y_val = y_val[mask_val].astype(int)

    # Fill any remaining nans (shouldn't be any)
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    # --- DIAGNOSTICS ---
    st.write("----DATA DIAGNOSTICS----")
    st.write("X_train shape:", X_train.shape)
    st.write("y_train shape:", y_train.shape)
    st.write("X_train columns:", X_train.columns.tolist())
    st.write("X_train dtypes:", X_train.dtypes)
    st.write("y_train dtype:", y_train.dtype)
    st.write("First 10 of y_train:", y_train.head(10))
    st.write("Unique values in y_train:", pd.unique(y_train))
    st.write("Any NaN in X_train?", X_train.isnull().any().any())
    st.write("Any NaN in y_train?", pd.isnull(y_train).any())
    st.write("Any inf in X_train?", np.isinf(X_train.to_numpy()).any())
    st.write("Any inf in y_train?", np.isinf(y_train.to_numpy()).any())
    st.write("X_train and y_train shapes match:", X_train.shape[0] == y_train.shape[0])

    # --- Model Training ---
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_val)
    acc_log = accuracy_score(y_val, y_pred_log)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)

    # --- Submission ---
    final_predictions = rf.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": final_predictions
    })

    report = f"""
Model Performance Report - Titanic
----------------------------------
Logistic Regression Accuracy: {acc_log:.4f}
Random Forest Accuracy:       {acc_rf:.4f}
Final model used:             Random Forest (n_estimators=100)
"""

    st.subheader("🌲 Feature Importances (Random Forest)")
    feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig5, ax5 = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax5)
    st.pyplot(fig5)

    # --- Download ZIP ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("submission.csv", submission.to_csv(index=False))
        zip_file.writestr("report.txt", report)

    st.subheader("📦 Download All Results")
    st.download_button(
        label="Download ZIP (submission + report)",
        data=zip_buffer.getvalue(),
        file_name="titanic_results.zip",
        mime="application/zip"
    )

    st.subheader("📄 Report Summary")
    st.code(report)

else:
    st.info("👋 Please upload both the training and test CSV files to get started.")

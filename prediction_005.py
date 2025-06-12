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

# Title
st.title("🚢 Titanic Survival Prediction + EDA")

# Upload CSV files
train_file = st.file_uploader("Upload train.csv", type=["csv"])
test_file = st.file_uploader("Upload test.csv", type=["csv"])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Combine datasets for preprocessing
    combine = [train, test]
    for dataset in combine:
        # Normalize column names
        dataset.columns = dataset.columns.str.strip().str.lower()

        # Fill missing age
        if 'age' in dataset.columns:
            dataset['age'].fillna(dataset['age'].median(), inplace=True)

        # Fill missing embarked
        if 'embarked' in dataset.columns:
            mode = dataset['embarked'].mode()
            if not mode.empty:
                dataset['embarked'].fillna(mode[0], inplace=True)
            else:
                dataset['embarked'].fillna('s', inplace=True)

        # Map embarked safely
        dataset['embarked'] = dataset['embarked'].astype(str).str.lower().map({'s': 0, 'c': 1, 'q': 2})
        dataset['embarked'].fillna(-1, inplace=True)
        dataset['embarked'] = dataset['embarked'].astype(int)

        # Map gender safely
        if 'sex' in dataset.columns:
            dataset['sex'] = dataset['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
            dataset['sex'].fillna(-1, inplace=True)
            dataset['sex'] = dataset['sex'].astype(int)

    # Fix fare in test
    if 'fare' in test.columns:
        test['fare'].fillna(test['fare'].median(), inplace=True)

    # Save PassengerId for output
    test_passenger_id = test['passengerid']

    # Drop unused columns
    train = train.drop(['name', 'ticket', 'cabin', 'passengerid'], axis=1)
    test = test.drop(['name', 'ticket', 'cabin', 'passengerid'], axis=1)

    # Show preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(train.head())

    # EDA visualizations
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

    # Train/Validation split
    X = train.drop("survived", axis=1)
    y = train["survived"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure no NaN/inf in features and labels
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = y_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_val)
    acc_log = accuracy_score(y_val, y_pred_log)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)

    # Final predictions
    final_predictions = rf.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": final_predictions
    })

    # Report
    report = f"""
    Model Performance Report - Titanic
    ----------------------------------
    Logistic Regression Accuracy: {acc_log:.4f}
    Random Forest Accuracy:       {acc_rf:.4f}
    Final model used:             Random Forest (n_estimators=100)
    """

    # Feature importance
    st.subheader("🌲 Feature Importances (Random Forest)")
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig5, ax5 = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax5)
    st.pyplot(fig5)

    # Prepare ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("submission.csv", submission.to_csv(index=False))
        zip_file.writestr("report.txt", report)

    # Download
    st.subheader("📦 Download All Results")
    st.download_button(
        label="Download ZIP (submission + report)",
        data=zip_buffer.getvalue(),
        file_name="titanic_results.zip",
        mime="application/zip"
    )

    # Report summary
    st.subheader("📄 Report Summary")
    st.code(report)

else:
    st.info("👋 Please upload both the training and test CSV files to get started.")
    st.code(report)

else:
    st.info("👋 Please upload both the training and test CSV files to get started.")

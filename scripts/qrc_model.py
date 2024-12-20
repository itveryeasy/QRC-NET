import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def load_dataset():
    """
    Fetch and preprocess the German Credit dataset from UCI repository.
    """
    # Fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    # Map target to binary (Good = 1, Bad = 0)
    y = y.map({1: 1, 2: 0})

    return X, y

def train_model(features, target):
    """
    Train a classical machine learning model on the features.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate recall
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    return recall

def run_qrc_model():
    """
    Orchestrates the QRC process.
    """
    features, target = load_dataset()
    recall = train_model(features, target)
    return recall

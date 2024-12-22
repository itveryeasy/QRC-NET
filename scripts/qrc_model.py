import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    """
    Load and preprocess the German Credit dataset.
    """
    column_names = [f"feature_{i}" for i in range(1, 25)] + ["Risk"]
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)
    data['Risk'] = data['Risk'].map({1: 1, 2: 0})
    return data

def train_model(features, target):
    """
    Train a classical machine learning model on the features.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    recall = recall_score(y_test, predictions)
    return recall

def run_qrc_model(dataset_path):
    """
    Orchestrates the QRC process.
    """
    data = load_dataset(dataset_path)
    features = data.drop(columns=['Risk'])
    target = data['Risk']
    recall = train_model(features, target)
    return recall

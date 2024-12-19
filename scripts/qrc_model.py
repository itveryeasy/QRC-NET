import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

def load_dataset(german.data-numeric):
    """
    Load the dataset and preprocess.
    """
    data = pd.read_csv(german.data-numeric, delim_whitespace=True, header=None)
    # Preprocessing example: Assuming binary classification.
    data['Risk'] = data['Risk'].map({'good': 1, 'bad': 0})
    features = data.drop(columns=['Risk'])
    target = data['Risk']
    return features, target

def train_model(features, target):
    """
    Train a classical machine learning model on extracted features.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    recall = recall_score(y_test, predictions)
    return recall

def run_qrc_model(dataset_path):
    """
    Orchestrates the QRC process.
    """
    features, target = load_dataset(dataset_path)
    recall = train_model(features, target)
    return recall

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset():
    """
    Load and preprocess the German Credit dataset from a local CSV file.
    """
    dataset_path = 'dataset/GermanCredit.csv'
    data = pd.read_csv(dataset_path)

    # Clean up column names by stripping spaces and converting to lowercase
    data.columns = data.columns.str.strip().str.lower()

    # Print the column names to verify
    print("Column Names:", data.columns)

    target_column = 'credit_risk'  # Updated target column name
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Encode categorical features if necessary
    features = pd.get_dummies(features, drop_first=True)

    return features, target


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
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate recall
    from sklearn.metrics import recall_score
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    return recall

def run_qrc_model():
    """
    Orchestrates the QRC process with the local dataset.
    """
    features, target = load_dataset()
    recall = train_model(features, target)
    return recall

# Run the model
recall_value = run_qrc_model()
print(f"Recall Score: {recall_value}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from imblearn.over_sampling import SMOTE

def load_dataset():
    """
    Load and preprocess the German Credit dataset from a local CSV file.
    """
    # Path to the dataset
    dataset_path = r'C:\Users\ASUS\QRC-NET\dataset\GermanCredit.csv'

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Assuming 'credit_risk' is the target column (Good = 1, Bad = 0)
    target_column = 'credit_risk'  # Update with actual target column name if different
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Encode categorical features if necessary
    features = pd.get_dummies(features, drop_first=True)

    return features, target

def train_model(features, target):
    """
    Train a classical machine learning model on the features.
    """
    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    features_balanced, target_balanced = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_balanced, target_balanced, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Adjust the threshold
    threshold = 0.4  # Adjust to balance precision and recall
    y_pred = (y_probs >= threshold).astype(int)

    # Evaluate recall
    recall = recall_score(y_test, y_pred)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return recall

def run_qrc_model():
    """
    Orchestrates the QRC process with the local dataset.
    """
    # Load dataset
    features, target = load_dataset()

    # Train model and calculate recall
    recall = train_model(features, target)
    return recall

# Run the model
if __name__ == "__main__":
    recall_value = run_qrc_model()
    print(f"Recall Score: {recall_value}")

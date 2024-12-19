import pandas as pd

def load_and_preprocess_data(german.data-numeric):
    """
    Load and preprocess the German Credit dataset.
    """
    # Define column names based on german.doc
    column_names = [f"feature_{i}" for i in range(1, 25)] + ["Risk"]

    # Load the numeric dataset
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

    # Map the target column to binary (1 = Good, 0 = Bad)
    data['Risk'] = data['Risk'].map({1: 1, 2: 0})

    return data

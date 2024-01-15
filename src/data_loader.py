import pandas as pd

def load_data(filepath):
    """Loads DNA sequence data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} sequences from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None

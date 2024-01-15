from src.data_loader import load_data
from src.feature_extractor import build_kmer_features
from src.model import train_model

def main():
    """Main pipeline for DNA sequence classification."""
    
    # 1. Load data
    df = load_data('data/dummy_dna_data.csv')
    if df is None:
        return
        
    sequences = df['sequence'].tolist()
    labels = df['class'].values

    # 2. Extract features
    X, vectorizer = build_kmer_features(sequences, kmer_size=4)

    # 3. Train model
    model = train_model(X, labels)

    # In a real app, you would save the model and vectorizer here
    print("\nPipeline finished.")

if __name__ == "__main__":
    main()


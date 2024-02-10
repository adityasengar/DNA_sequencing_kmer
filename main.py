import argparse
import joblib
import os
from src.data_loader import load_data
from src.feature_extractor import build_kmer_features
from src.model import train_model

def main():
    """Main pipeline for DNA sequence classification."""
    parser = argparse.ArgumentParser(description="DNA Sequence Classifier using k-mers")
    parser.add_argument("data_file", type=str, help="Path to the input CSV data file.")
    parser.add_argument("--kmer_size", type=int, default=4, help="Size of the k-mers to use for feature extraction.")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], default='train', help="Mode to run: 'train' or 'predict'.")
    parser.add_argument("--model_dir", type=str, default='models', help="Directory to save/load the model and vectorizer.")
    args = parser.parse_args()

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, 'model.pkl')
    vectorizer_path = os.path.join(args.model_dir, 'vectorizer.pkl')
    
    # 1. Load data
    df = load_data(args.data_file)
    if df is None:
        return

    if args.mode == 'train':
        print("\n--- Training Mode ---")
        sequences = df['sequence'].tolist()
        labels = df['class'].values

        # 2. Extract features
        X, vectorizer = build_kmer_features(sequences, kmer_size=args.kmer_size)

        # 3. Train model
        model = train_model(X, labels)

        # 4. Save model and vectorizer
        print(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        print(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        
        print("\nTraining pipeline finished.")

    elif args.mode == 'predict':
        print("\n--- Prediction Mode ---")
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Error: Model or vectorizer not found in {args.model_dir}. Please run in 'train' mode first.")
            return

        # 1. Load model and vectorizer
        print("Loading model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # 2. Prepare new data
        sequences = df['sequence'].tolist()
        kmer_words = [' '.join(getKmers(seq, args.kmer_size)) for seq in sequences]
        X_new = vectorizer.transform(kmer_words)

        # 3. Make predictions
        print("Making predictions...")
        predictions = model.predict(X_new)
        df['predicted_class'] = predictions
        
        print("Predictions:")
        print(df[['sequence', 'predicted_class']])

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def getKmers(sequence, k=6):
    """Function to generate k-mers from a DNA sequence."""
    return [sequence[x:x+k] for x in range(len(sequence) - k + 1)]

def main():
    """Main function to run the DNA sequence analysis."""
    # Load data
    try:
        df = pd.read_csv('data/dummy_dna_data.csv')
    except FileNotFoundError:
        print("Error: data/dummy_dna_data.csv not found. Please place your data there.")
        return

    # Feature Extraction (k-mer counting)
    df['words'] = df.apply(lambda x: getKmers(x['sequence']), axis=1)
    df = df.drop('sequence', axis=1)

    kmer_texts = [' '.join(words) for words in df['words']]
    
    cv = CountVectorizer(ngram_range=(4,4)) # Using 4-mers as an example
    X = cv.fit_transform(kmer_texts)
    
    y = df.iloc[:, 0].values

    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training a Multinomial Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()

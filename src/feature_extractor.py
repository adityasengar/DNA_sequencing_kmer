from sklearn.feature_extraction.text import CountVectorizer

def getKmers(sequence, k=6):
    """Generates k-mers from a DNA sequence."""
    return [sequence[x:x+k] for x in range(len(sequence) - k + 1)]

def build_kmer_features(sequences, kmer_size=4):
    """Builds a feature matrix from a list of DNA sequences using k-mer counts."""
    print(f"Extracting {kmer_size}-mer features...")
    
    # Apply the k-mer function to each sequence
    kmer_words = [' '.join(getKmers(seq, kmer_size)) for seq in sequences]
    
    # Vectorize the k-mer words
    vectorizer = CountVectorizer(ngram_range=(kmer_size, kmer_size))
    X = vectorizer.fit_transform(kmer_words)
    
    print(f"Feature matrix created with shape: {X.shape}")
    return X, vectorizer

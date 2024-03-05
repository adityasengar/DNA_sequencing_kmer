# DNA Sequencing k-mer Classifier

This project provides a command-line tool to classify DNA sequences into gene families using a machine learning approach based on k-mer frequency analysis.

The original analysis was performed in a Jupyter Notebook and has been refactored into a structured and reusable Python application.

## Project Overview

The tool performs the following steps:
1.  **Loads** DNA sequences and their corresponding classes from a CSV file.
2.  **Extracts Features** by breaking down each DNA sequence into k-mers (substrings of length k) and counting their frequencies.
3.  **Trains** a Multinomial Naive Bayes classifier on the k-mer frequency features.
4.  **Saves** the trained classifier and the feature vectorizer for later use.
5.  **Predicts** the class of new DNA sequences using the pre-trained model.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/DNA_sequencing_kmer.git
    cd DNA_sequencing_kmer
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The application is controlled via `main.py` and has two modes: `train` and `predict`. You will need a CSV file with 'sequence' and 'class' columns. A dummy example is provided in `data/dummy_dna_data.csv`.

### Training the Model

To train the model on your dataset and save the trained artifacts:

```bash
python main.py data/your_dna_data.csv --mode train --kmer_size 6
```

This will:
- Load data from `data/your_dna_data.csv`.
- Build features using 6-mers.
- Train the classifier.
- Save the trained model to `models/model.pkl` and the vectorizer to `models/vectorizer.pkl`.

### Making Predictions

Once the model is trained, you can use it to predict the classes of new sequences.

```bash
python main.py data/new_sequences.csv --mode predict --kmer_size 6
```

This will:
- Load the pre-trained model and vectorizer from the `models/` directory.
- Load the new sequences.
- Print the predicted class for each sequence.

### Command-Line Arguments

-   `data_file`: Path to the input CSV data file. (Required)
-   `--kmer_size`: The size of the k-mer to use for feature extraction. (Default: 4)
-   `--mode`: `train` or `predict`. (Default: `train`)
-   `--model_dir`: Directory to save/load the model and vectorizer. (Default: `models`)
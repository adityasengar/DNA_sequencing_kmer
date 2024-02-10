from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_and_evaluate(X, y):
    """Trains a Multinomial Naive Bayes model and evaluates it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on test set: {accuracy:.4f}")
    
    return model

def predict(model, X):
    """Makes predictions on new data."""
    return model.predict(X)
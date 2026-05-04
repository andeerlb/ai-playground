# This code is a complete implementation of a simple spam detection model using the Naive Bayes algorithm.
# It loads a dataset of SMS messages, preprocesses the data, trains a model, evaluates its performance, and saves the model for future use.
# The dataset is expected to be in a TSV (tab-separated values) format with two columns: 'label' (ham or spam) and 'message' (the text of the SMS).
# Spam means the message is unwanted or unsolicited
# ham abreviates "human" and is used to indicate that the message is not spam.

import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# LOAD DATA
file_path = "data/sms.tsv"
data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

print("=== DATA INFO ===")
print("Total samples:", len(data))
print("\nLabel distribution:")
print(data['label'].value_counts())

# PREPROCESS
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# TEXT -> NUMBERS
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN
model = MultinomialNB()
model.fit(X_train, y_train)

# SAVE MODEL

# PKL is a common format for saving Python objects, especially in machine learning
# It allows you to save the trained model and vectorizer so you can load them later without retraining
joblib.dump(model, "model.pkl")
# Saving the vectorizer is important because it contains the mapping of words to feature indices
# When you load the model later, you'll also need the vectorizer to transform new messages into the same feature space
# Vectorizer is not a model but a transformer, so we save it separately
# Example of vectorizer: it might have a vocabulary like {'free': 0, 'money': 1, 'now': 2, ...}
joblib.dump(vectorizer, "vectorizer.pkl")

# Predict is used to get the class labels (0 or 1) for the test set, while predict_proba gives the probabilities of each class
predictions = model.predict(X_test)
# Predict proba returns an array of shape (n_samples, n_classes) with the probabilities for each class.
probabilities = model.predict_proba(X_test)

print("\n=== EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# SAMPLE PREDICTIONS
print("\n=== SAMPLE PREDICTIONS ===")
for i in range(5):
    print("\nMessage:", data['message'].iloc[i])
    print("Real:", y.iloc[i])
    print("Predicted:", model.predict(X[i])[0])
    print("Spam probability:", probabilities[i][1]) # probabilities[i][1] gives the probability of the message being spam (class 1)

# CUSTOM TESTS
print("\n=== CUSTOM TESTS ===")

tests = [
    "Free money now!!!",
    "Hey, are we meeting today?",
    "Congratulations, you won a prize!",
    "Call me when you arrive",
    "URGENT! You have won a 1 week FREE membership!"
]

for msg in tests:
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    proba = model.predict_proba(msg_vec)[0][1]

    print(f"\nMessage: {msg}")
    print(f"Prediction: {pred} (1=spam, 0=not spam)")
    print(f"Spam probability: {proba:.4f}")
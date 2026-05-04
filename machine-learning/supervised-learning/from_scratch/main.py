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

# Label distribution is important to check if the dataset is balanced or imbalanced. An imbalanced dataset can lead to a model that performs well on the majority class but poorly on the minority class.
# How it works: value_counts() counts the occurrences of each unique value in the 'label' column, giving us insight into how many messages are labeled as 'ham' and how many are labeled as 'spam'.
print("\nLabel distribution:")
print(data['label'].value_counts())

# PREPROCESS
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# TEXT -> NUMBERS
# CountVectorizer converts a collection of text documents to a matrix of token counts. It creates a vocabulary of all the unique words in the dataset and transforms each message into a vector where each element represents the count of a specific word in that message.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# SPLIT
# train_test_split is a function from scikit-learn that splits arrays or matrices into random train and test subsets. The test_size parameter specifies the proportion of the dataset to include in the test split (in this case, 20%), and random_state ensures that the split is reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN
# MultinomialNB is a Naive Bayes classifier for multinomial models, which is suitable for classification with discrete features (like word counts). It calculates the probability of each class based on the frequency of words in the training data and uses these probabilities to make predictions.
# Naive Bayes is a simple yet effective algorithm for text classification tasks like spam detection, and it often performs well even with small datasets.
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
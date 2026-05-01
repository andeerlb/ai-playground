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
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# EVALUATE
predictions = model.predict(X_test)

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
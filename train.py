# train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)


print("Dataset Loaded")
print(df.head())


# ==============================
# 2. Data Cleaning
# ==============================

# Keep only required columns

df = df[['v1', 'v2']]


# Rename columns

df.columns = [
    "label",
    "message"
]


# Remove missing values

df.dropna(
    inplace=True
)


# Convert labels

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)


print("\nAfter Cleaning:")
print(df.head())


# ==============================
# 3. Split Data
# ==============================

X = df["message"]

y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 4. TF-IDF Feature Extraction
# ==============================

vectorizer = TfidfVectorizer(
    stop_words="english"
)


X_train_vector = vectorizer.fit_transform(
    X_train
)


X_test_vector = vectorizer.transform(
    X_test
)


print("Text Converted into Numerical Features")


# ==============================
# 5. Train Model
# ==============================

model = MultinomialNB()


model.fit(
    X_train_vector,
    y_train
)


print("Model Training Completed")


# ==============================
# 6. Evaluation
# ==============================

y_pred = model.predict(
    X_test_vector
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nAccuracy:")
print(accuracy)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==============================
# 7. Save Model
# ==============================

joblib.dump(
    model,
    "spam_model.pkl"
)


joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


print("\nModel Saved Successfully!")

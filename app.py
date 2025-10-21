import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ------------------------------
# Load & preprocess dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    st.write("🔍 Columns found:", df.columns.tolist())

    # Rename if columns differ
    if 'text' not in df.columns or 'label' not in df.columns:
        # Try to auto-detect text/label columns
        possible_text_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower()]
        possible_label_cols = [c for c in df.columns if 'label' in c.lower() or 'category' in c.lower()]
        df.rename(columns={possible_text_cols[0]: 'text', possible_label_cols[0]: 'label'}, inplace=True)
    
        # Drop missing or empty rows
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'].str.strip() != ""]

    # If dataset has 'spam'/'ham', convert — otherwise, skip
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    df = df[df['label'].notna()]
    return df

data = load_data()

# ------------------------------
# Split & Train Model
# ------------------------------
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    return model, vectorizer, acc

model, vectorizer, acc = train_model(data)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📧 Spam Email Classifier (Enron Dataset)")
st.markdown("### Identify whether an email is **Spam** or **Not Spam** 🕵️‍♂️")

st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Accuracy:** {acc*100:.2f}%")
st.sidebar.write("**Model:** Logistic Regression + TF-IDF")
st.sidebar.write("**Dataset:** Enron Spam Corpus")

# ------------------------------
# Email Input
# ------------------------------
user_input = st.text_area("✉️ Paste your email text here:")

if st.button("🔍 Analyze Email"):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        proba = model.predict_proba(input_tfidf)[0][1]

        if prediction == 1:
            st.error(f"🚨 This email is **Spam** (Probability: {proba:.2%})")
        else:
            st.success(f"✅ This email is **Not Spam** (Probability: {1-proba:.2%})")
    else:
        st.warning("Please enter some email text first!")

# ------------------------------
# Model Performance (Optional)
# ------------------------------
if st.checkbox("Show Model Performance Report"):
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    st.text(classification_report(y_test, y_pred))

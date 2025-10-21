This Streamlit web app uses Machine Learning (TF-IDF + Logistic Regression) to classify whether an email is Spam or Not Spam based on its text content.
It’s trained on the Enron Spam Corpus, one of the most well-known real-world email datasets.

🚀 Built with simplicity and clarity in mind — ideal for learning text classification, feature extraction, and ML app deployment.

✨ Features

✅ Upload or paste any email text
✅ Detect Spam vs Not Spam in real time
✅ Shows prediction probability
✅ Display detailed classification metrics (accuracy, precision, recall, F1)
✅ Built with Streamlit, Scikit-learn, and TF-IDF Vectorization

📂 Project Structure
📁 spam-email-classifier/
│
├── app.py                # Main Streamlit app
├── combined_data.csv     # Dataset (Enron Spam Corpus or combined CSV)
├── requirements.txt      # Dependencies for Hugging Face / local run
└── README.md             # Project documentation

⚙️ Installation & Setup
🧩 1. Clone the Repository
git clone https://github.com/<your-username>/spam-email-classifier.git
cd spam-email-classifier

🧠 2. Create Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

📦 3. Install Dependencies

Create a file named requirements.txt:

streamlit
pandas
scikit-learn


Then install:

pip install -r requirements.txt

▶️ Run the App Locally
streamlit run app.py


Then open the app in your browser at:
👉 http://localhost:8501

🌐 Deploy on Hugging Face Spaces
Step 1: Create a new Space

Go to Hugging Face Spaces

Click New Space → select Streamlit → set it to Public

Step 2: Upload Files

Upload the following files:

app.py

combined_data.csv

requirements.txt

README.md

Step 3: Done! 🎉

Hugging Face will automatically build and deploy your app.

📊 Example Output

Input:

"Congratulations! You've won a $1000 gift card. Click the link below to claim your prize."

Output:

🚨 This email is Spam (Probability: 98.34%)

📈 Model Details
Component	Description
Vectorizer	TF-IDF (max 5000 features, English stop words)
Model	Logistic Regression
Accuracy	~95–98% (depends on dataset split)
Dataset	Enron Spam Corpus (combined_data.csv)
🧾 Sample Requirements.txt
streamlit
pandas
scikit-learn

💡 Future Improvements

Add custom dataset upload option

Show confusion matrix visualization

Add NLP preprocessing (lemmatization, stemming)

Save and reuse trained model with pickle

👩‍💻 Author

Bano Rani
📍 Built with ❤️ using Python, Streamlit, and Scikit-learn
🔗 GitHub Profile: 

🔗 Hugging Face Space https://huggingface.co/spaces/bano1/ML_Spam_classifier

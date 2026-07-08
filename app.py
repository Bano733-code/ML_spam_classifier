import streamlit as st
import joblib


st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧"
)


st.title("📧 Email Spam Classification")

st.write(
    "Enter an email message to check whether it is spam or not."
)


@st.cache_resource
def load_model():

    model = joblib.load(
        "spam_model.pkl"
    )

    vectorizer = joblib.load(
        "vectorizer.pkl"
    )

    return model, vectorizer



model, vectorizer = load_model()



email = st.text_area(
    "Enter Email Text"
)



if st.button("Check Email"):


    if email:

        email_vector = vectorizer.transform(
            [email]
        )


        prediction = model.predict(
            email_vector
        )


        probability = model.predict_proba(
            email_vector
        )


        if prediction[0] == 1:

            st.error(
                "🚨 This Email is Spam"
            )

        else:

            st.success(
                "✅ This Email is Not Spam"
            )


        st.write(
            "Confidence:"
        )

        st.write(
            f"{max(probability[0])*100:.2f}%"
        )

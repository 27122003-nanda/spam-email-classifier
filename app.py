import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI Title
st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it is Spam or Not Spam.")

# Input box
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚨 This message is *SPAM*.")
        else:
            st.success("✔ This message is *NOT SPAM*.")
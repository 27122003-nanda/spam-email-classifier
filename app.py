import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it is Spam or Not Spam.")

# Rule-based spam keywords
spam_keywords = [
    "lottery", "winner", "won", "prize", "free", "gift", "money", "cash",
    "credit card", "loan", "urgent", "claim", "click here", "offer",
    "bonus", "guaranteed", "limited time", "congratulations"
]

def rule_based_spam(message):
    msg = message.lower()
    for word in spam_keywords:
        if word in msg:
            return True
    return False

# Input box
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1️⃣ Rule-based detection first
        if rule_based_spam(message):
            st.error("⚠️ This message is *SPAM*.")
        else:
            # 2️⃣ ML Model Prediction
            transformed = vectorizer.transform([message])
            result = model.predict(transformed)[0]

            if result == 1:
                st.error("⚠️ This message is *SPAM*.")
            else:
                st.success("✓ This message is *NOT SPAM*.")

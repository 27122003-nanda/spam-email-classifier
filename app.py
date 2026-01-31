import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Email Classifier")

st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it is Spam or Not Spam.")

# Spam keyword rules (manual filter)
spam_keywords = [
    "win", "winner", "prize", "free", "gift", "money", "cash",
    "lottery", "urgent", "time", "click here", "offer",
    "credit", "card", "loan", "limited time", "congratulations",
    "guaranteed", "verification", "update your account", "kyc",
    "blocked", "pay immediately"
]

# Input box
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠ Please enter a message.")
    else:
        # Rule-based spam detection
        msg_lower = message.lower()
        if any(word in msg_lower for word in spam_keywords):
            st.error("⚠ *This message is SPAM.* (Rule-based detection)")
        else:
            # ML prediction
            transformed = vectorizer.transform([message])
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.error("⚠ *This message is SPAM.*")
            else:
                st.success("✓ This message is NOT SPAM.")

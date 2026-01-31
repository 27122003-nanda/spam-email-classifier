import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Email Classifier")

st.title("📩 Spam Email Classifier")
st.write("Enter a message to check if it is Spam or Not Spam.")

# Input box
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform message
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ This message is SPAM.")
        else:
            st.success("✔️ This message is NOT SPAM.")

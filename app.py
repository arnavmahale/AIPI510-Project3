import streamlit as st
import requests

API_URL = "https://sst2-api-664742743732.us-east1.run.app/predict"

st.title("Sentiment Classifier for Text")
st.write("Enter a sentence and get a sentiment prediction from our model.")

text_input = st.text_area("Enter text:", height=100)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(API_URL, json={"text": text_input})
        if response.status_code == 200:
            data = response.json()
            label = "Positive 😄" if data["label"] == 1 else "Negative 😡"
            st.subheader(label)
            st.write(f"Model probability: **{data['probability']:.4f}**")
        else:
            st.error("API error. Check backend logs.")

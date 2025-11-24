import os

import httpx
import streamlit as st


st.set_page_config(page_title="SST-2 Sentiment", page_icon="💬", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict(text: str) -> tuple[int, float] | None:
    url = f"{API_URL.rstrip('/')}/predict"
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, json={"text": text})
        resp.raise_for_status()
        data = resp.json()
    return int(data["label"]), float(data["probability"])


def main() -> None:
    st.title("SST-2 Sentiment")
    st.caption(f"API endpoint: {API_URL}")

    text = st.text_area("Enter text", height=150, placeholder="I loved this movie!")

    if st.button("Predict", type="primary", disabled=not text.strip()):
        with st.spinner("Calling API..."):
            try:
                label, prob = predict(text)
            except Exception as e:
                st.error(f"Request failed: {e}")
                return

        sentiment = "Positive" if label == 1 else "Negative"
        st.success(f"{sentiment} ({prob:.2%} confidence)")


if __name__ == "__main__":
    main()

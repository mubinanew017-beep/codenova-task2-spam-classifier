
import streamlit as st
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    data_path = Path(__file__).parent / "SMSSpamCollection"

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    return model

st.title("📩 SMS Spam Detector")
st.write(
    "Enter an SMS message below to check whether "
    "it is spam or a legitimate message."
)

try:
    model = load_model()

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Example: Congratulations! You won a prize...",
        height=150
    )

    if st.button("Check Message", type="primary"):
        if message.strip():
            prediction = model.predict([message])[0]

            if prediction == "spam":
                st.error("⚠️ This message is likely SPAM.")
            else:
                st.success("✅ This message is likely HAM (legitimate).")
        else:
            st.warning("Please enter a message first.")

    st.caption("Educational project — predictions may not always be correct.")

except Exception as e:
    st.error("The app could not load the dataset.")
    st.info(
        "Please check that SMSSpamCollection is present "
        "in the same GitHub repository as app.py."
    )

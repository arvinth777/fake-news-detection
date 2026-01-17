import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model (adjust path/name)
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the directory.")
    st.stop()

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.markdown("Enter a news article to check its authenticity")

article = st.text_area("Paste article text here:", height=200)

if st.button("Analyze Article"):
    if article:
        # Preprocess and predict
        try:
            features = vectorizer.transform([article])
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0]
            
            # Notebook trained Real as 1, Fake as 0.
            if prediction == 0: 
                st.error(f"⚠️ FAKE NEWS ({confidence[0]:.1%} confidence)")
            else:
                st.success(f"✅ REAL NEWS ({confidence[1]:.1%} confidence)")
        except Exception as e:
             st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text!")

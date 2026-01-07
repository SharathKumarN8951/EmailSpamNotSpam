import streamlit as st
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Email Spam vs Not Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("multinomial_nb_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("📩Email Spam vs Not Spam Classifier")
st.write("Enter a email message to check whether it is **Spam** or **Not Spam**")

text = st.text_area(
    "✍️ Enter your message",
    height=150,
    placeholder="Type your message here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Vectorize input text
        text_vector = vectorizer.transform([text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Handle string labels (ham / spam)
        if prediction == "ham":
            st.success("✅ Prediction: **Not Spam (Ham)**")
        elif prediction == "spam":
            st.error("🚫 Prediction: **Spam**")
        else:
            st.info(f"Prediction: {prediction}")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<hr><p style='text-align:center;font-size:13px;'>Developed By Sharath_Kumar_N & Built with ❤️ using Streamlit  &  Scikit-learn</p>",
    unsafe_allow_html=True
)

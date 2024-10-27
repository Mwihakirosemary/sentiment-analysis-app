import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your models
@st.cache_data
def load_models():
    naive_bayes_model = joblib.load('models/naive_bayes_model (2).pkl')
    logistic_regression_model = joblib.load('models/logistic_regression_model (2).pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer (2).pkl')
    return naive_bayes_model, logistic_regression_model, svm_model, vectorizer

naive_bayes_model, logistic_regression_model, svm_model, vectorizer = load_models()

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    text = " ".join(lemmatized_words)
    return text

# Prediction function
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    prediction = model.predict(text_features)[0]
    return prediction

# Streamlit UI Enhancements
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Sentiment Analysis App 🎉</h1>", unsafe_allow_html=True)

# Text Input with Icon
st.markdown("### Enter Text for Sentiment Analysis 💬")
user_input = st.text_input("")

# Model Selection
st.markdown("### Choose a Sentiment Model 🧠")
model_choice = st.selectbox("Pick your model:", [
    "Naive Bayes 🤖 - Fast and efficient",
    "Logistic Regression 📈 - Reliable and interpretable",
    "Support Vector Machine 🛠️ - Powerful with large datasets"
])

# Instruction Section
st.markdown("### Instructions:")
st.info("1. Type your text in the box above.\n2. Choose a model for sentiment analysis.\n3. Hit 'Analyze Sentiment' to see the results!")

# Button to trigger prediction
if st.button("Analyze Sentiment 🔍"):
    if user_input:
        if model_choice.startswith("Naive Bayes"):
            result = predict_sentiment(user_input, naive_bayes_model, vectorizer)
        elif model_choice.startswith("Logistic Regression"):
            result = predict_sentiment(user_input, logistic_regression_model, vectorizer)
        elif model_choice.startswith("Support Vector Machine"):
            result = predict_sentiment(user_input, svm_model, vectorizer)
        else:
            result = "Invalid model choice"
        
        # Display the prediction result
        st.success("Analysis complete!")
        st.write(f"**Prediction:** {result}")
    else:
        st.write("Please enter some text.")

# Background and Layout Styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #fbc2eb, #a6c1ee);
    }
    </style>
    """,
    unsafe_allow_html=True
)
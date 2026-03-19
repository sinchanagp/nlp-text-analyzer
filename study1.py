import streamlit as st
import nltk
import spacy
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import pandas as pd

# -----------------------------
# Downloads
# -----------------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Pragmatics
# -----------------------------
def pragmatics_analysis(text):
    text_lower = text.lower()

    if "?" in text:
        return "❓ Question / Request"
    elif any(word in text_lower for word in ["please", "could you", "can you"]):
        return "🙏 Polite Request"
    elif any(word in text_lower for word in ["urgent", "immediately", "asap"]):
        return "⚡ Urgent Message"
    elif any(word in text_lower for word in ["bad", "poor", "worst", "not happy"]):
        return "😡 Negative Feedback"
    elif any(word in text_lower for word in ["good", "great", "excellent", "happy"]):
        return "😊 Positive Feedback"
    else:
        return "ℹ️ Informative Statement"

# -----------------------------
# Voice Input
# -----------------------------
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Could not recognize speech"

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Advanced NLP Analyzer", layout="wide")

st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
}
.box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Advanced NLP Analyzer</div>', unsafe_allow_html=True)

# -----------------------------
# INPUT OPTIONS
# -----------------------------
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input type:",
    ["Text Input", "🎤 Voice Input", "📁 Upload File"]
)

text = ""

if input_method == "Text Input":
    text = st.text_area("Enter your sentence:")

elif input_method == "🎤 Voice Input":
    if st.button("Record Voice"):
        text = voice_input()
        st.success(f"Recognized Text: {text}")

elif input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.success("File loaded successfully!")

# -----------------------------
# ANALYSIS
# -----------------------------
if st.button("🚀 Analyze"):

    if text.strip():

        st.subheader("Input Text")
        st.write(text)

        tokens = word_tokenize(text)

        col1, col2 = st.columns(2)

        # Tokenization
        with col1:
            st.markdown("### 🔹 Tokens")
            st.write(tokens)

        # Stemming
        with col2:
            ps = PorterStemmer()
            stemmed = [ps.stem(w) for w in tokens]
            st.markdown("### 🔹 Stemming")
            st.write(stemmed)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
        st.markdown("### 🔹 Lemmatization")
        st.write(lemmatized)

        # POS Tagging (Colored)
        st.markdown("### 🔹 POS Tagging (Colored)")
        pos_tags = pos_tag(tokens)

        pos_html = ""
        for word, tag in pos_tags:
            color = "#2ecc71" if "NN" in tag else "#3498db" if "VB" in tag else "#e67e22"
            pos_html += f"<span style='color:{color}; font-weight:bold'>{word}({tag})</span> "

        st.markdown(pos_html, unsafe_allow_html=True)

        # Dependency Parsing
        st.markdown("### 🔹 Dependency Parsing")
        doc = nlp(text)
        dep_data = [(token.text, token.dep_, token.head.text) for token in doc]
        df = pd.DataFrame(dep_data, columns=["Word", "Dependency", "Head"])
        st.dataframe(df)

        # Named Entity Recognition
        st.markdown("### 🔹 Named Entities")
        if doc.ents:
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            st.write(ents)
        else:
            st.write("No entities found")

        # Pragmatics
        st.markdown("### 🔹 Pragmatics")
        st.success(pragmatics_analysis(text))

    else:
        st.warning(" Please provide input!")
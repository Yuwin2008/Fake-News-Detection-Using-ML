import streamlit as st
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton>button {
    width:100%;
    height:3em;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

footer {
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/calibrated_svm_model.pkl"
)

# ---------------------------------
# NLP OBJECTS
# ---------------------------------

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()

# ---------------------------------
# PREPROCESS FUNCTION
# ---------------------------------

def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = []

    for word in text.split():

        if word not in stop_words:

            words.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(words)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title(" About")

st.sidebar.info("""
### Fake News Detector

Model:
- Calibrated Linear SVM

Features:
- TF-IDF Vectorization
- 20,000 Features

Performance:
- F1 Score: 99.02%
- CV Accuracy: 98.87%
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Try Example Articles")

real_example = """
The Federal Reserve announced that interest rates
would remain unchanged this month according to Reuters.
"""

fake_example = """
Scientists confirm that drinking coffee grants
telepathic powers and allows communication with aliens.
"""

if st.sidebar.button("🟢 Load Real Example"):
    st.session_state["example_text"] = real_example

if st.sidebar.button("🔴 Load Fake Example"):
    st.session_state["example_text"] = fake_example

with st.sidebar.expander("Model Details"):

    st.write("""
    Dataset Size:
    39,105 articles

    Domain:
    Political News

    Technologies:
    - Python
    - Scikit-Learn
    - Streamlit
    """)

# ---------------------------------
# HERO SECTION
# ---------------------------------

st.markdown("""
# 📰 Fake News Detection using AI

### Detect misinformation using Machine Learning and NLP

Built using:

- TF-IDF Vectorization
- Support Vector Machines
- Streamlit

---
""")

# ---------------------------------
# LAYOUT
# ---------------------------------

col1, col2 = st.columns([3,1])

with col1:

    text = st.text_area(
    "Paste News Article Here",
    value=st.session_state.get(
        "example_text",
        ""
    ),
    height=300
)

with col2:

    st.metric(
        "Dataset Size",
        "39,105"
    )

    st.metric(
        "F1 Score",
        "99.02%"
    )

    st.metric(
        "Model",
        "Calibrated SVM"
    )

# ---------------------------------
# PREDICTION
# ---------------------------------

if st.button("Predict"):

    cleaned_text = preprocess(text)

    vectorized_text = tfidf.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        vectorized_text
    )[0]

    probabilities = model.predict_proba(
        vectorized_text
    )[0]

    if prediction == 1:

        confidence = probabilities[1] * 100

        st.success(
            f"🟢 REAL NEWS ({confidence:.2f}%)"
        )

    else:

        confidence = probabilities[0] * 100

        st.error(
            f"🔴 FAKE NEWS ({confidence:.2f}%)"
        )

    st.subheader("Prediction Confidence")

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "Real News Probability",
            f"{probabilities[1]*100:.2f}%"
        )

        st.progress(
            float(probabilities[1])
        )

    with col2:

        st.metric(
            "Fake News Probability",
            f"{probabilities[0]*100:.2f}%"
        )

        st.progress(
            float(probabilities[0])
        )

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.caption("""
Built by GodofThunder2407(RL Yuwin)

Machine Learning • NLP • Streamlit
""")

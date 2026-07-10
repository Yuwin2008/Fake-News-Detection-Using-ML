#  Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Linear SVM](https://img.shields.io/badge/Model-Linear%20SVM-purple)
![F1 Score](https://img.shields.io/badge/F1%20Score-99.02%25-brightgreen)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success)](https://fake-news-detection-using-ml.streamlit.app/)

##  Project Overview

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Real** or **Fake**.

The project explores the complete NLP workflow including:

* Data preprocessing
* Text cleaning
* Exploratory Data Analysis (EDA)
* TF-IDF vectorization
* Model benchmarking
* Cross-validation
* Model selection

The final model uses **TF-IDF Vectorization** combined with a **Linear Support Vector Machine (SVM)** classifier.

---

##  Dataset

The dataset consists of two CSV files:

* `Fake.csv`
* `True.csv`

The datasets were merged and labeled as:

* `0 → Fake News`
* `1 → Real News`

Total dataset size:

* **39,105 news articles**

---

##  Preprocessing Pipeline

The following preprocessing steps were applied:

* Convert text to lowercase
* Remove URLs
* Remove punctuation
* Remove numbers
* Remove stopwords
* Remove Reuters boilerplate text
* Tokenization
* Lemmatization

---

##  Exploratory Data Analysis

The following analyses were performed:

* Class distribution visualization
* Article length distribution
* Word clouds
* Top 20 words in real news
* Top 20 words in fake news

---

##  Feature Engineering

Text data was converted into numerical representations using **TF-IDF Vectorization** with:

* Unigrams
* Bigrams
* Maximum feature limit of 20,000

---

##  Models Evaluated

| Model                   | Precision  | Recall     | F1 Score   |
| ----------------------- | ---------- | ---------- | ---------- |
| Multinomial Naive Bayes | 95.72%     | 93.98%     | 94.85%     |
| Random Forest           | 97.50%     | 98.51%     | 98.01%     |
| Logistic Regression     | 98.06%     | 98.84%     | 98.45%     |
| Linear SVM              | 98.66%     | 99.27%     | 98.97%     |
| Caliberated SVM         | **99.01%** | **99.03%** | **99.02%** |

---

##  Final Model

**Caliberated Support Vector Machine (SVM)** achieved the best performance.

Final configuration:

* TF-IDF Vectorizer
* Linear SVM Classifier

Cross-validation performance:

* **CV Mean Accuracy:** 98.87%
* **CV Standard Deviation:** 0.0021

---

##  Example Predictions

### Example 1

Input:

NASA successfully launched a new Mars rover mission yesterday.

Prediction:

🔴 FAKE NEWS (90.33%)

---

### Example 2

Input:

Scientists confirm that secret lizard people control global governments.

Prediction:

🔴 FAKE NEWS (95.26%)

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* NLTK
* WordCloud

---

##  Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection-Using-ML.git
cd Fake-News-Detection-Using-ML
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, install the required packages manually:

```bash
pip install streamlit
pip install scikit-learn
pip install nltk
pip install pandas
pip install numpy
pip install joblib
```

---

### 4. Download NLTK Resources

Run Python:

```bash
python
```

Then execute:

```python
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

Exit Python:

```python
exit()
```

---

### 5. Verify Model Files

Ensure the following files exist inside the `models` folder:

```text
models/
│
├── tfidf_vectorizer.pkl
└── svm_fake_news.pkl
```

---

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

If it does not, open:

```text
http://localhost:8501
```

---

##  Features

* Fake News Detection using Machine Learning
* TF-IDF Vectorization
* Calibrated Linear SVM Classifier
* Prediction Confidence Scores
* Probability Visualization
* Interactive Streamlit UI
* Example News Articles
* Modern Sidebar Dashboard

---

##  Note

This model was trained primarily on political news articles and performs best on articles from similar domains.

Predictions on topics such as:

* Space
* Science
* Entertainment
* Sports
* Technology

may be less reliable due to domain shift.


---

## Limitations

This model was trained primarily on political news articles and performs best on articles from similar domains.

Predictions on topics such as:
- Science
- Space
- Sports
- Entertainment
- Technology

may be less reliable due to domain shift.

---


## 📈 Future Improvements

* Confidence score visualization
* Explainable AI techniques (SHAP/LIME)
* Transformer-based models (BERT)
* Live news article classification

---

##  License
**MIT License**
This project is intended for educational and research purposes.

---

##  Author

**GodofThunder(RL Yuwin)**

Machine Learning Enthusiast | Kaggle Competitor | Aspiring AI Engineer

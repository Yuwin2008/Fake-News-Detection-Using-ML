#  Fake News Detection using Machine Learning

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

## 🏆 Final Model

**Caliberated Support Vector Machine (SVM)** achieved the best performance.

Final configuration:

* TF-IDF Vectorizer
* Linear SVM Classifier

Cross-validation performance:

* **CV Mean Accuracy:** 98.87%
* **CV Standard Deviation:** 0.0021

---

## 🚀 Example Predictions

### Example 1

Input:

NASA successfully launched a new Mars rover mission yesterday.

Prediction:

🟢 REAL NEWS ()

---

### Example 2

Input:

Scientists confirm that secret lizard people control global governments.

Prediction:

🔴 FAKE NEWS (95.26%)

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* NLTK
* WordCloud

---

## 📈 Future Improvements

* Streamlit deployment
* Confidence score visualization
* Explainable AI techniques (SHAP/LIME)
* Transformer-based models (BERT)
* Live news article classification

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**RL Yuwin**

Machine Learning Enthusiast | Kaggle Competitor | Aspiring AI Engineer

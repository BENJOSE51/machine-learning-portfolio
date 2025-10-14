# NLP TF-IDF Project

## Overview
This project explores a text dataset and builds a classification model using TF-IDF vectorization.
The notebook contains EDA, preprocessing, feature extraction with TF-IDF, model training, evaluation, and interpretation.

## Contents
- `notebooks/` — `04_NLP_TFIDF_Project_Final.ipynb`
- `data/` — text dataset(s) used in the project
- `models/` — saved model artifacts (e.g., `nlp_tfidf_pipeline_v1.joblib`)

## Highlights
- Cleaned and preprocessed text (lowercase, stopword removal, tokenization, optional lemmatization)
- TF-IDF vectorization with tuned n-gram and max-features settings
- Model evaluation: accuracy, precision, recall, F1-score, and confusion matrix
- Feature importance: top TF-IDF tokens per class

## Next Steps
- Deploy via FastAPI and Streamlit.
- Consider transformer-based models (Hugging Face) for better accuracy on complex datasets.
- Add model explainability for text predictions.

## Tech Stack
Python, pandas, scikit-learn, nltk/spacy, joblib

---

# **Fake News Detection**

This repository contains a Jupyter Notebook that demonstrates how to classify news articles as fake or real using machine learning techniques. The dataset used in this project was sourced from a Kaggle competition.

---

## **Overview**

The spread of fake news has become a significant issue in recent years, impacting public opinion and decision-making. This project aims to build a machine learning model to detect fake news articles based on their content. The model uses natural language processing (NLP) techniques to preprocess the text data and logistic regression for classification.

The dataset includes features such as the article title, author, and text, with a target variable indicating whether the news is fake (`1`) or real (`0`).

---

## **Dataset**

- **Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- **Features**:
  - `id`: Unique identifier for each news article.
  - `title`: The title of the news article.
  - `author`: The author of the news article.
  - `text`: The main content of the news article.
  - `label`: Target variable (1 = Fake, 0 = Real).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`train.csv`) is loaded into a Pandas DataFrame.
2. **Data Preprocessing**:
   - Text data is cleaned using regular expressions to remove special characters and numbers.
   - Stopwords are removed using NLTK's stopword list.
   - Words are stemmed using the Porter Stemmer to reduce them to their root forms.
3. **Feature Extraction**:
   - TF-IDF Vectorizer is used to convert text data into numerical features suitable for machine learning models.
4. **Model Training**:
   - Logistic Regression is used as the classification model.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Accuracy score is calculated to evaluate the model's performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- pandas
- numpy
- nltk
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/FakeNewsDetection.git
   cd FakeNewsDetection
   ```

2. Ensure that the dataset file (`train.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Fake-News-Detect.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The logistic regression model provides an accuracy score that indicates its performance in detecting fake news articles. Further improvements can be made by experimenting with other machine learning models or advanced NLP techniques.

---

## **Acknowledgments**

- The dataset was sourced from [Kaggle](https://www.kaggle.com/c/fake-news/data).
- Special thanks to the developers of NLTK and Scikit-learn for providing robust NLP and machine learning libraries.

---

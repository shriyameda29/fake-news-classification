# Fake News Classification

## Overview
This project analyzes linguistic and structural differences between fake and real 
news articles using the ISOT Fake News Dataset. It builds and compares four machine 
learning models to classify news articles as fake or real using NLP techniques 
including TF-IDF and Word2Vec.

## Dataset
This project uses the ISOT Fake News Dataset, available on Kaggle:
https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

Download 'Fake.csv' and 'True.csv' and place them in a folder called 'data/' 
in the project directory.

Sample data files ('Fake_sample.csv' and 'True_sample.csv') are included 
in the 'data/' folder for reference. To run the full project, download 
'Fake.csv' and 'True.csv' from Kaggle and place them in the 'data/' folder.

## Project Structure
fakenews_classifier_proj/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── setup_db.ipynb              # Database setup and data ingestion
├── news.db                     # SQL Database file
├── eda.ipynb                   # Exploratory data analysis
├── preprocessing_classification.ipynb  # Preprocessing, feature extraction, and classification
├── requirements.txt            # Required modules
└── README.md                   # Project documentation

## Setup Instructions
1. Clone or download the project folder
2. Install dependencies: pip install -r requirements.txt
3. Download the ISOT dataset from Kaggle and place 'Fake.csv' and 'True.csv' in the 'data/' folder
4. Run the notebooks in order:
   - 'setup_db.ipynb'
   - 'eda.ipynb'
   - 'preprocessing_classification.ipynb'

## Example Usage
Sample datasets 'Fake_sample.csv' and 'True_sample.csv' have been provided in 'data/' folder to load into project
In 'setup_db.ipynb', instead of running:
'''
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")
'''
replace CSV file names with sample CSV file names, then run rest of notebooks (in aforementioned order)

The 'predict_article' function in 'preprocessing_classification.ipynb' can classify any news article as fake or real:

'''
title = "Your article title here"
text = "Your article text here"
predict_article(title, text)
'''

## Models and Results
Four models were trained and evaluated:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.9949 | 0.9950 |
| TF-IDF + Naive Bayes | 0.9500 | 0.9513 |
| Word2Vec + Logistic Regression | 0.9748 | 0.9753 |
| Word2Vec + Gaussian Naive Bayes | 0.8876 | 0.8889 |

TF-IDF + Logistic Regression performed best overall.

## Requirements
- Python 3.12.3
- See 'requirements.txt' for full dependency list
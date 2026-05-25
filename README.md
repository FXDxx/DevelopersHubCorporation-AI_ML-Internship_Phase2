***DevelopersHubCorporation-AI_ML-Internship_Phase2***

Task 1: News Topic Classifier Using BERT

Details

Problem statement: classify news headlines into topic categories.

Dataset: From huggingface: ag_news

Model: BERT

Evaluation metrics: loss, accuracy, f1-score

Summary: This model takes input articles from news sources and predicts the news topic from the given text. 


Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

Details
Problem statement: A reusable and production-ready machine learning pipeline for predicting customer
churn.

Dataset: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Models: LogisticRegression

Evaluation metrics: Accuracy with improvement from GridSearchCV hyperparameter tuning

Summary: A re-usable pipeline for predicting customer churn. Accuracy measure is used to evaluate the performance of the model and hyperparameter tuning done using GridSearchCV.

Task 3: Auto Tagging Support Tickets Using LLM
Details
Problem statement: Automatically tag support tickets into categories

Dataset: Generated my own sample data

Models: For zero-shot classification: facebook/bart-large-mnli, Few shot: google/flan-t5-small, For fine-tuning: distilbert-base-uncased. 
Download model: https://drive.google.com/drive/folders/1_jsI_BwVDiBwEjMje1z_R8j1gN2NvZci?usp=drive_link

Evaluation metrics: on the basis of results from model doing fine-tuning.

Summary: From the given text predict the tag it is representing in context. For this a sample data is taken and applied zero-shot and few-shot classification and then fine-tuned by other model. The results were evaluated in comparison from fine-tuned model results.

Consumer Complaint Classification (NLP Multi-Class Project)


📘 Project Overview

This project performs multi-class text classification on consumer complaints data from the Consumer Financial Protection Bureau (CFPB)
.
Each complaint is classified into categories such as:

Credit Reporting

Debt Collection

Loan

Mortgage

The goal is to use NLP techniques and machine learning to predict the complaint type.

⚙️ Steps Performed

1️⃣ Exploratory Data Analysis (EDA)

Inspected dataset structure and missing values

<img width="526" height="382" alt="image" src="https://github.com/user-attachments/assets/2402e69f-8187-4d28-baa4-6fdb4e904dd6" />


Visualized class imbalance

<img width="971" height="378" alt="image" src="https://github.com/user-attachments/assets/95afc0ff-c8fb-4b05-b2d2-4145cb8839fe" />


Applied Stratified Down Sampling for class balance


2️⃣ Text Preprocessing

Lowercasing

Removing punctuation, numbers, and stopwords

Tokenization

Creating a clean text_clean column



3️⃣ Feature Extraction

Used TF-IDF Vectorization

Generated feature matrix for machine learning models

4️⃣ Model Training

Trained the following multi-class models:

Logistic Regression

Multinomial Naive Bayes

Linear Support Vector Classifier (SVC)

📸 Screenshot:


5️⃣ Model Comparison
Model	Accuracy	F1-Score
Logistic Regression	0.89	0.88
Linear SVC	0.88	0.87
MultinomialNB	0.84	0.83

<img width="670" height="569" alt="image" src="https://github.com/user-attachments/assets/a411d717-f8d6-402f-bbb0-aedfcbbe4d20" />



6️⃣ Prediction

Tested the model with new complaint examples:

predict_category([
  "My credit report shows incorrect information",
  "Debt collector is calling repeatedly",
  "Mortgage lender raised my interest rate suddenly"
])


✅ Output:

→ Credit Reporting
→ Debt Collection
→ Mortgage


<img width="917" height="539" alt="image" src="https://github.com/user-attachments/assets/222abb93-8298-423d-9c79-2c2314268d47" />



🧠 Concepts Used

Natural Language Processing (NLP)

Feature Engineering (TF-IDF)

Multi-Class Classification

Model Evaluation (Accuracy, F1-Score, Confusion Matrix)

Python, Pandas, Scikit-Learn, NLTK, Seaborn, Matplotlib

🚀 How to Run the Project

Open Google Colab

Upload the notebook: consumer_complaint_classifier.ipynb

Upload dataset (balanced CSV)

Run all cells sequentially (1 → 6)

Observe printed metrics and charts

Save screenshots with your name & timestamp

💾 Dependencies

If needed, install required packages in Colab:

!pip install scikit-learn nltk seaborn matplotlib wordcloud xgboost

🧾 References

Dataset: Consumer Complaint Database - data.gov

Libraries: Scikit-learn, NLTK, Seaborn, Matplotlib

Platform: Google Colab


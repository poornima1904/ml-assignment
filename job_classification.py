import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
from collections import Counter

## Load the dataset
file_name = 'new_data_set.csv'  # Replace with your file name
data = pd.read_csv(file_name)

## Dataset Preparation
X = data['Job Description']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=1,
    max_df=0.85
)


X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train an SVC model
base_model = LinearSVC(random_state=42, class_weight='balanced')
base_model.fit(X_train_vec, y_train)

# Prediction on Testing Data Set
y_pred = base_model.predict(X_test_vec)
results_df = pd.DataFrame({'Job Description': X_test, 'Predicted Category': y_pred})
print(results_df)
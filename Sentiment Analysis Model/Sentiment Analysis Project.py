import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. SETUP & DATA GENERATION
# ==========================================
# Ensure necessary NLTK data is downloaded
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Mock Dataset (In a real scenario, you would load this via pd.read_csv('reviews.csv'))
data = {
    'review': [
        "I absolutely loved this product! It works perfectly.",
        "Terrible experience, the item arrived broken.",
        "Great value for money, highly recommended.",
        "Worst purchase I ever made. Do not buy.",
        "It's okay, does the job but nothing special.",
        "Fantastic quality and fast shipping!",
        "Customer service was rude and unhelpful.",
        "I am very happy with my purchase.",
        "Not worth the price, very cheap material.",
        "Five stars! Will definitely buy again.",
        "Awful, simply awful.",
        "Decent product, but delivery was slow."
    ],
    'sentiment': [
        "Positive", "Negative", "Positive", "Negative", "Neutral", 
        "Positive", "Negative", "Positive", "Negative", "Positive", 
        "Negative", "Neutral"
    ]
}

df = pd.DataFrame(data)
print(f"\nDataset Head:\n{df.head()}")

# ==========================================
# 2. PREPROCESSING
# ==========================================
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

print("\nPreprocessing reviews...")
df['clean_review'] = df['review'].apply(preprocess_text)

# ==========================================
# 3. VECTORIZATION (TF-IDF)
# ==========================================
# Convert text to numbers using Term Frequency-Inverse Document Frequency
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 4. MODEL TRAINING (Naive Bayes)
# ==========================================
print("Training Naive Bayes Model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. VISUALIZATION
# ==========================================
# Plot Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Sentiment Analysis Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# 7. REAL-TIME PREDICTION DEMO
# ==========================================
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

print("\n--- Live Test ---")
sample_review = "SO bad"
print(f"Review: '{sample_review}' -> Sentiment: {predict_sentiment(sample_review)}")
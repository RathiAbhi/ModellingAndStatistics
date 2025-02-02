import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 📌 Sample Dataset (Spam vs Ham)
data = {
    "message": ["Win a lottery now", "Call me tomorrow", "Urgent! Claim your prize",
                "Meeting at 5 PM", "Congratulations! You won", "Let's go for lunch"],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# 📌 Convert Labels to Numeric
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 📌 Convert Text to Numerical Vectors (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Multinomial Naïve Bayes Model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# 📌 Predict
y_pred = mnb.predict(X_test)

# 📌 Evaluate Model
print("Multinomial Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
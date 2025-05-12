# 🛠️ Install library
!pip install nltk

# 📚 Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📥 Download NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 📂 Load Dataset
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# 🏷️ Add labels
fake_df['label'] = 0
real_df['label'] = 1

# 🧮 Combine and shuffle
df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)

# 🔍 Keep only title and text
df['content'] = df['title'] + " " + df['text']
X = df['content']
y = df['label']

# 🧹 Preprocessing (remove stopwords)
def clean_text(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

X = X.apply(clean_text)

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔤 TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🧠 Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 🔍 Evaluate
y_pred = model.predict(X_test_vec)

print("\n📈 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

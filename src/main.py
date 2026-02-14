import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. Setup & Data Loading ---
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / "dataset"

folders = ["politics", "sports"]
docs = []
labels = []

print("Reading files...")

for label, folder_name in enumerate(folders):
    path = data_dir / folder_name  # Using Pathlib for cleaner paths
    
    if not path.exists():
        print(f"Skipping {folder_name}, path not found at {path}")
        continue

    for file in os.listdir(path):
        if file.endswith(".txt"):
            try:
                with open(path / file, 'r', encoding='utf-8', errors='ignore') as f:
                    docs.append(f.read())
                    labels.append(label)
            except Exception as e:
                print(f"Error reading {file}: {e}")

if len(docs) == 0:
    print("Error: No documents loaded. Please check your dataset directory structure.")
    exit()

print(f"Loaded {len(docs)} files total.")

# --- 2. Vectorization & Splitting ---
# Tfidf: ngrams=(1,2) captures "ball" and "tennis ball"
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# --- 3. Model Definition ---
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(dual='auto', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# --- 4. Training & Accuracy Comparison ---
print("\n" + "="*30)
print("MODEL ACCURACY COMPARISON")
print("="*30)

# Dictionary to store accuracy scores for potential ranking later
model_scores = {}

for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    model_scores[name] = acc
    print(f"{name:<15}: {acc*100:.2f}%")

# --- 5. Prediction on Sample Text for EACH Model ---
def predict_text(text, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return folders[pred[0]]

#Test Samples
test_politics = "The committee debated the new housing regulations for several hours before reaching a consensus. Critics argue the law doesn't go far enough to address the crisis."

test_sports = "After a slow start to the season, the team rallied to win five consecutive games. The captain's performance in the playoffs was instrumental in securing the trophy."

#Run Predictions
print("\n" + "="*30)
print("FINAL TEST RESULTS")
print("="*30)

print(f"Text: {test_politics}")
for name, clf in classifiers.items():
    print(f"{name:<15}: {predict_text(test_politics, clf)}")

print("-" * 30)

print(f"Text: {test_sports}")
for name, clf in classifiers.items():
    print(f"{name:<15}: {predict_text(test_sports, clf)}")
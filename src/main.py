import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#get the parent directory (project root)
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / "dataset"

folders = ["politics", "sports"]

docs = []
labels = []

print("Reading files...")

#loading data
for label, folder_name in enumerate(folders):
    path = os.path.join(data_dir, folder_name)
    
    #quick check to avoid crashing if folder is missing
    if not os.path.exists(path):
        print(f"Skipping {folder_name}, path not found.")
        continue

    for file in os.listdir(path):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
                    docs.append(f.read())
                    labels.append(label)
            except:
                pass #just skip bad files

print(f"Loaded {len(docs)} files total.")

#using TF-IDF because it handles common words better than simple counts
#ngrams=(1,2) means we look at single words AND pairs (like "white house" or "match point")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

#80-20 split is standard
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

#let's compare these three
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(dual='auto'), #SVM is usually great for text
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\nResults:")
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc*100:.2f}% accuracy")

#simple test function to try it out
def predict_text(text, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return folders[pred[0]]

#Example usage with the last trained model (Random Forest)
print("\nQuick Test:")
sample = "The senator voted against the new bill yesterday."
print(f"Text: {sample}")
print(f"Prediction: {predict_text(sample, classifiers['Random Forest'])}")
import os
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#set up paths and load the data
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / "dataset"

#define the categories we're working with
categories = ["politics", "sports"]
documents = []
category_labels = []

print("Reading files...")

#loop through each category folder and load the text files
for label, folder_name in enumerate(categories):
    folder_path = data_dir / folder_name
    
    if not folder_path.exists():
        print(f"Skipping {folder_name}, path not found at {folder_path}")
        continue

    #read all .txt files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                file_path = folder_path / filename
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    documents.append(file.read())
                    category_labels.append(label)
            except Exception as error:
                print(f"Error reading {filename}: {error}")

#check if we loaded any documents
if len(documents) == 0:
    print("Error: No documents loaded. Please check your dataset directory structure.")
    sys.exit()

print(f"Loaded {len(documents)} files total.")

#prepare the data with TF-IDF vectorization and split into train/test sets
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_features = tfidf_vectorizer.fit_transform(documents)

#split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_features, category_labels, test_size=0.2, random_state=42
)

#define the classifiers we'll compare
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(dual='auto', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

#train each model and print accuracy
print("\n" + "*"*30)
print("MODEL ACCURACY COMPARISON")
print("*"*30)

for model_name, model in models.items():
    #fit the model on training data
    model.fit(X_train, y_train)
    #make predictions on test data
    predictions = model.predict(X_test)
    #calculate and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name:<15}: {accuracy*100:.2f}%")

#function to predict the category for a given text
def classify_text(text, model):
    #transform the text using the fitted vectorizer
    text_vector = tfidf_vectorizer.transform([text])
    #get the prediction
    category_index = model.predict(text_vector)
    return categories[category_index[0]]

#collect samples to test
test_samples = []

#check for command-line input file
if len(sys.argv) > 1:
    input_filename = sys.argv[1]
    input_file_path = Path(input_filename)
    
    if input_file_path.exists():
        print(f"\nReading input from file: {input_filename}")
        try:
            with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as input_file:
                content = input_file.read().strip()
                if content:
                    test_samples.append(content)
                else:
                    print("Warning: File is empty.")
        except Exception as error:
            print(f"Error reading file: {error}")
    else:
        print(f"\n[ERROR] File '{input_filename}' not found.")
else:
    print("\nNo input file provided. Using default test samples.")
    #Adding some default examples if no file is given
    test_samples.append("The committee debated the new housing regulations for several hours before reaching a consensus.")
    test_samples.append("After a slow start to the season, the team rallied to win five consecutive games.")

#printing the prediction results
if test_samples:
    print("\n" + "*"*30)
    print("PREDICTION RESULTS")
    print("*"*30)

    for index, sample_text in enumerate(test_samples):
        #print preview of the text
        preview = (sample_text[:60] + '...') if len(sample_text) > 60 else sample_text
        
        print(f"\nSample {index+1}: \"{preview}\"")
        print("-" * 30)
        
        #predict with each model
        for model_name, model in models.items():
            predicted_category = classify_text(sample_text, model)
            print(f"{model_name:<15}: {predicted_category}")
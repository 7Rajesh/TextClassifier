# Text Classification: Politics vs Sports

Project that classifies text documents into two categories: Politics and Sports using TF-IDF vectorization and multiple classification algorithms.

## Features
- TF-IDF vectorization
- Comparison of three classifiers:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Easy to extend for additional categories

## Installation
Prerequisites
Python 3.7 or higher
pip (Python package manager)


## How to Run

Follow these steps to set up and execute the project.

### 1. Install Dependencies
This project requires `scikit-learn` for the machine learning models. Install it via pip:

```bash
pip install scikit-learn
```
### 2.Clone the Repository
```bash
git clone https://github.com/7Rajesh/TextClassifier.git
cd TextClassifier
```
## 3.How To Run
Run the classifier from the project root:
IF No Text input file is provided it will show result for default sample text
```bash
python src/main.py
```

IF input file is provided
```bash
python src/main.py sample.txt
```
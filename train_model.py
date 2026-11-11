
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Downloading phishing URL dataset...")

# Download the UCI PhiUSIIL dataset
dataset = fetch_ucirepo(id=967)

X = dataset.data.features
y = dataset.data.targets

print("Available URL columns:", X.columns.tolist())
print("Available target columns:", y.columns.tolist())

# Find URL and label columns
url_col = next(c for c in X.columns if c.lower() == "url")
label_col = next(c for c in y.columns if c.lower() == "label")

# Prepare the data
data = pd.DataFrame({
    "url": X[url_col].astype(str),
    "label": pd.to_numeric(y[label_col], errors="coerce")
})

data = data.dropna()
data = data.drop_duplicates(subset="url")
data = data[data["label"].isin([0, 1])]

print("Total URLs:", len(data))

# Dataset labels: 0 = phishing, 1 = legitimate
X_train, X_test, y_train, y_test = train_test_split(
    data["url"],
    data["label"].astype(int),
    test_size=0.2,
    random_state=42,
    stratify=data["label"].astype(int)
)

print("Training the machine learning model...")

# Character-level URL pattern learning
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )
    )
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save trained model
joblib.dump(model, "phishing_url_model.joblib")

print("\nModel saved as phishing_url_model.joblib")
print("Training completed!")

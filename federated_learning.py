
import re
import ipaddress
import joblib
import numpy as np
import pandas as pd

from urllib.parse import urlparse
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------------
# 1. Extract numerical features from a URL
# -----------------------------------------

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "digit_ratio",
    "special_char_ratio",
    "uses_https",
    "has_ip_address",
    "subdomain_count",
    "has_at_symbol",
    "has_suspicious_keyword",
    "has_punycode",
    "has_port",
    "has_fragment"
]


def extract_features(raw_url):
    url = str(raw_url).strip()

    candidate = url if "://" in url else "https://" + url
    parsed = urlparse(candidate)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    digits = sum(char.isdigit() for char in url)
    special = sum(
        not char.isalnum() for char in url
    )

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    keywords = [
        "login", "verify", "account", "secure",
        "update", "bank", "password", "kyc"
    ]

    suspicious = int(
        any(word in url.lower() for word in keywords)
    )

    subdomains = max(0, len(host.split(".")) - 2)

    return [
        np.log1p(len(url)),
        np.log1p(len(host)),
        np.log1p(len(path)),
        np.log1p(len(query)),
        url.count("."),
        url.count("-"),
        digits / max(len(url), 1),
        special / max(len(url), 1),
        int(parsed.scheme.lower() == "https"),
        has_ip,
        subdomains,
        int("@" in url),
        suspicious,
        int("xn--" in host),
        int(parsed.port is not None),
        int(bool(parsed.fragment))
    ]


# -----------------------------------------
# 2. Load the phishing URL dataset
# -----------------------------------------

print("Downloading dataset...")

dataset = fetch_ucirepo(id=967)

X_data = dataset.data.features
y_data = dataset.data.targets

url_column = next(
    col for col in X_data.columns
    if col.lower() == "url"
)

label_column = next(
    col for col in y_data.columns
    if col.lower() == "label"
)

data = pd.DataFrame({
    "url": X_data[url_column].astype(str),
    "label": pd.to_numeric(
        y_data[label_column], errors="coerce"
    )
})

data = data.dropna()
data = data[data["label"].isin([0, 1])]
data = data.drop_duplicates(subset="url")

print("Total URLs:", len(data))


# Dataset labels:
# 0 = phishing
# 1 = legitimate
#
# Convert to:
# 1 = phishing
# 0 = legitimate

y = (data["label"].to_numpy() == 0).astype(int)

print("Extracting URL features...")

X = np.array(
    [extract_features(url) for url in data["url"]],
    dtype=np.float64
)


# -----------------------------------------
# 3. Split, scale and distribute the data
# -----------------------------------------

# Keep test data separate for final evaluation
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create a validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

# Standardize numerical features using training data only
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

feature_std[feature_std < 1e-8] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

# Shuffle training data before distributing it
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_train))

X_train = X_train[indices]
y_train = y_train[indices]

# Three simulated clients
client_X = np.array_split(X_train, 3)
client_y = np.array_split(y_train, 3)

print("\nThree clients created:")
for i in range(3):
    print(f"Client {i + 1}: {len(client_X[i])} training URLs")
# -----------------------------------------
# 4. Logistic regression model
# -----------------------------------------

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def local_training(X_local, y_local, weights, bias):
    """Train one client using its local data."""

    learning_rate = 0.05
    local_epochs = 3
    regularization = 0.001

    w = weights.copy()
    b = float(bias)

    for _ in range(local_epochs):
        predictions = sigmoid(X_local @ w + b)

        error = predictions - y_local

        dw = (X_local.T @ error) / len(y_local)
        dw += regularization * w

        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


# -----------------------------------------
# 5. Federated Averaging (FedAvg)
# -----------------------------------------

print("\nStarting Federated Learning...")

global_weights = np.zeros(X_train.shape[1])
global_bias = 0.0

rounds = 10

for round_number in range(rounds):

    client_weights = []
    client_biases = []
    client_sizes = []

    for client_number in range(3):

        local_w, local_b = local_training(
            client_X[client_number],
            client_y[client_number],
            global_weights,
            global_bias
        )

        client_weights.append(local_w)
        client_biases.append(local_b)
        client_sizes.append(len(client_X[client_number]))

    # Weighted FedAvg aggregation
    total_size = sum(client_sizes)

    global_weights = sum(
        client_weights[i] * client_sizes[i]
        for i in range(3)
    ) / total_size

    global_bias = sum(
        client_biases[i] * client_sizes[i]
        for i in range(3)
    ) / total_size

    print(f"Federated round {round_number + 1}/{rounds} completed")


# -----------------------------------------
# 6. Select threshold and evaluate model
# -----------------------------------------

from sklearn.metrics import precision_recall_curve

# Select threshold using validation data
val_probabilities = sigmoid(
    X_val @ global_weights + global_bias
)

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nSelected threshold:", round(float(best_threshold), 4))

# Evaluate only once on the separate test set
probabilities = sigmoid(
    X_test @ global_weights + global_bias
)

predictions = (probabilities >= best_threshold).astype(int)

print("\n========== FEDERATED MODEL RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[1, 0],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[1, 0]))
joblib.dump(
    {
        "weights": global_weights,
        "bias": global_bias,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": float(best_threshold),
        "description": "Three-client FedAvg URL feature model"
    },
    "federated_phishing_model.joblib"
)
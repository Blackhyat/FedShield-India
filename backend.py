
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from pathlib import Path
import ipaddress
import re
import numpy as np
import joblib


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="FedShield-India API",
    description="Federated Learning + Explainable URL Analysis",
    version="0.4.0",
)

MODEL_PATH = Path(__file__).with_name(
    "federated_phishing_model.joblib"
)

# Fixed risk thresholds used by the frontend and backend
LOWER_RISK_MAX = 34
CAUTION_MAX = 69
HIGH_RISK_MIN = 70


# --------------------------------------------------
# LOAD FEDERATED LEARNING MODEL
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)

    required_keys = {
        "feature_mean",
        "feature_std",
        "weights",
        "bias",
    }

    if not isinstance(model, dict) or not required_keys.issubset(model):
        raise ValueError("Federated model has an unexpected format.")

    print("Federated phishing model loaded successfully!")

except Exception as error:
    model = None
    print("Could not load Federated model:", error)


# --------------------------------------------------
# REQUEST AND RESPONSE MODELS
# --------------------------------------------------

class URLRequest(BaseModel):
    url: str


class AnalysisResponse(BaseModel):
    risk: str
    score: int
    reasons: list[str]


# --------------------------------------------------
# URL FEATURE EXTRACTION
# Must match federated training feature order
# --------------------------------------------------

def extract_features(raw_url):

    url = str(raw_url).strip()

    candidate = (
        url if "://" in url else "https://" + url
    )

    parsed = urlparse(candidate)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    digits = sum(char.isdigit() for char in url)
    special = sum(not char.isalnum() for char in url)

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

    subdomains = max(
        0, len(host.split(".")) - 2
    )

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


# --------------------------------------------------
# MODEL PROBABILITY
# --------------------------------------------------

def sigmoid(value):

    value = np.clip(value, -30, 30)

    return 1 / (1 + np.exp(-value))


def predict_phishing_probability(raw_url):

    features = np.array(
        extract_features(raw_url),
        dtype=np.float64
    )

    feature_mean = np.asarray(
        model["feature_mean"],
        dtype=np.float64
    )

    feature_std = np.asarray(
        model["feature_std"],
        dtype=np.float64
    )

    weights = np.asarray(
        model["weights"],
        dtype=np.float64
    )

    # Avoid division by zero
    feature_std = np.where(
        feature_std == 0,
        1.0,
        feature_std
    )

    if (
        features.shape != feature_mean.shape
        or features.shape != feature_std.shape
        or features.shape != weights.shape
    ):
        raise ValueError("Federated model feature dimensions do not match.")

    features = (
        features - feature_mean
    ) / feature_std

    logit = float(
        features @ weights + model["bias"]
    )

    return float(sigmoid(logit))


# --------------------------------------------------
# URL VALIDATION
# --------------------------------------------------

def validate_url(raw_url):

    value = raw_url.strip()

    if not value or len(value) > 2048:
        raise HTTPException(
            status_code=400,
            detail="Enter a URL between 1 and 2048 characters."
        )

    if any(char.isspace() for char in value):
        raise HTTPException(
            status_code=400,
            detail="URL must not contain spaces."
        )

    candidate = (
        value if "://" in value else "https://" + value
    )

    try:

        parsed = urlparse(candidate)

        host = parsed.hostname or ""

        # Validate port syntax
        _ = parsed.port

        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Unsupported URL scheme")

        if not host:
            raise ValueError("Missing hostname")

        # Accept IP addresses
        try:
            ipaddress.ip_address(host)

        except ValueError:

            # Validate domain structure
            if "." not in host or len(host) > 253:
                raise ValueError("Invalid domain")

            labels = host.rstrip(".").split(".")

            for label in labels:

                if (
                    not label
                    or len(label) > 63
                    or not re.fullmatch(
                        r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?",
                        label
                    )
                ):
                    raise ValueError("Invalid domain label")

        return candidate, parsed, host

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Please enter a valid HTTP or HTTPS URL."
        )


# --------------------------------------------------
# FIXED RISK CLASSIFICATION
# --------------------------------------------------

def classify_risk(score):

    if score <= LOWER_RISK_MAX:
        return "Lower Risk"

    elif score <= CAUTION_MAX:
        return "Caution"

    return "High Risk"


# --------------------------------------------------
# EXPLAINABLE WARNING SIGNALS
# --------------------------------------------------

def get_warning_reasons(value, parsed, host):

    reasons = []

    if parsed.scheme.lower() == "http":
        reasons.append(
            "The URL uses HTTP rather than HTTPS."
        )

    if "@" in parsed.netloc:
        reasons.append(
            "The URL contains an @ symbol, which can hide the actual destination."
        )

    if len(value) > 100:
        reasons.append(
            "The URL is unusually long."
        )

    if host.count("-") >= 3:
        reasons.append(
            "The domain contains several hyphens."
        )

    keywords = [
        "verify", "login", "secure",
        "account", "kyc", "update",
        "password", "bank"
    ]

    if any(word in host.lower() for word in keywords):
        reasons.append(
            "The domain uses account or verification-related wording."
        )

    if len(host.split(".")) >= 5:
        reasons.append(
            "The domain has many subdomain levels."
        )

    try:
        ipaddress.ip_address(host)

        reasons.append(
            "The URL uses an IP address instead of a regular domain name."
        )

    except ValueError:
        pass

    if "xn--" in host.lower():
        reasons.append(
            "The domain contains an internationalized hostname."
        )

    if not reasons:
        reasons.append(
            "No additional rule-based warning signals were detected."
        )

    return reasons


# --------------------------------------------------
# MAIN URL ANALYSIS
# --------------------------------------------------

def analyze_url(raw_url: str) -> AnalysisResponse:

    candidate, parsed, host = validate_url(raw_url)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Federated model is not loaded. Check the model file."
        )

    try:

        probability = predict_phishing_probability(candidate)

        score = int(
            round(probability * 100)
        )

        score = max(0, min(100, score))

    except Exception:

        # Keep internal model details out of API responses
        print("Federated model prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to analyze this URL. Please try again."
        )

    # Apply fixed thresholds
    risk = classify_risk(score)

    # Explainable warning signals
    reasons = get_warning_reasons(
        raw_url.strip(),
        parsed,
        host
    )

    # Add model explanation first
    reasons.insert(
        0,
        f"The Federated Learning model estimates a {score}% phishing probability."
    )

    return AnalysisResponse(
        risk=risk,
        score=score,
        reasons=reasons
    )


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "FedShield-India",
        "status": "running",
        "version": "0.4.0",
        "mode": "federated machine learning + rule-based analysis",
        "model_loaded": model is not None,
        "risk_thresholds": {
            "lower_risk": "0-34%",
            "caution": "35-69%",
            "high_risk": "70-100%"
        }
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "version": "0.4.0"
    }


# --------------------------------------------------
# ANALYZE ENDPOINT
# --------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalysisResponse
)
def analyze(request: URLRequest):

    return analyze_url(request.url)

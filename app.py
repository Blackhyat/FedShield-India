
import streamlit as st
import requests
import sqlite3
import hashlib
import hmac
import secrets
import re
import json
import os
import pandas as pd

from pathlib import Path
from datetime import datetime


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FedShield-India | URL Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = Path(
    os.getenv(
        "FEDSHIELD_DB_PATH",
        str(BASE_DIR / "fedshield.db")
    )
)

API_URL = os.getenv(
    "FEDSHIELD_API_URL",
    "https://fedshield-backend.onrender.com/analyze"
)


# =========================================================
# 2. CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263449;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: linear-gradient(145deg, #17243a, #111827);
        border: 1px solid #293951;
        border-radius: 14px;
        padding: 21px;
        min-height: 125px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 800;
        margin-top: 10px;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 3px;
    }

    .panel {
        background: #111c2e;
        border: 1px solid #293951;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .risk-high {
        background: #3b1721;
        color: #fca5a5;
        padding: 13px 16px;
        border-radius: 9px;
        font-weight: 700;
    }

    .risk-caution {
        background: #3b2d13;
        color: #fcd34d;
        padding: 13px 16px;
        border-radius: 9px;
        font-weight: 700;
    }

    .risk-low {
        background: #12352c;
        color: #86efac;
        padding: 13px 16px;
        border-radius: 9px;
        font-weight: 700;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 12px;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #293951;
        border-radius: 10px;
    }

    hr {
        border-color: #293951;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                url TEXT NOT NULL,
                risk TEXT NOT NULL,
                score REAL NOT NULL,
                reasons TEXT,
                scanned_at TEXT NOT NULL
            )
        """)

        conn.commit()


initialize_database()


# =========================================================
# 4. AUTHENTICATION
# =========================================================

def valid_email(email):
    return bool(
        re.fullmatch(
            r"[^\s@]+@[^\s@]+\.[^\s@]+",
            email.strip()
        )
    )


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        310000
    ).hex()


def create_user(email, password):
    email = email.strip().lower()
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users
                (email, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                email,
                password_hash,
                salt,
                datetime.now().isoformat(timespec="seconds")
            ))
            conn.commit()

        return True

    except sqlite3.IntegrityError:
        return False


def verify_user(email, password):
    email = email.strip().lower()

    with get_connection() as conn:
        row = conn.execute("""
            SELECT password_hash, password_salt
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

    if not row:
        return False

    stored_hash, salt = row
    entered_hash = hash_password(password, salt)

    return hmac.compare_digest(stored_hash, entered_hash)


# =========================================================
# 5. SCAN HISTORY FUNCTIONS
# =========================================================

def save_scan(user_id, url, risk, score, reasons):
    if isinstance(reasons, list):
        reasons = json.dumps(reasons)
    elif not isinstance(reasons, str):
        reasons = str(reasons)

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO scan_history
            (user_id, url, risk, score, reasons, scanned_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            url,
            risk,
            float(score),
            reasons,
            datetime.now().astimezone().isoformat(timespec="seconds")
        ))
        conn.commit()


def get_user_history(user_id):
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT
                id,
                url,
                risk,
                score,
                reasons,
                scanned_at
            FROM scan_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, conn, params=(user_id,))

    return df


def get_user_statistics(user_id):
    df = get_user_history(user_id)

    if df.empty:
        return {
            "total": 0,
            "high_risk": 0,
            "caution": 0,
            "lower_risk": 0
        }

    risks = df["risk"].astype(str).str.lower()

    return {
        "total": len(df),
        "high_risk": risks.isin(
            ["high risk", "high"]
        ).sum(),
        "caution": risks.isin(
            ["caution", "medium risk", "medium"]
        ).sum(),
        "lower_risk": risks.isin(
            ["lower risk", "low risk", "low"]
        ).sum()
    }


def clear_user_history(user_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM scan_history WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()


# =========================================================
# 6. SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# 7. LOGIN AND REGISTRATION
# =========================================================


# =========================================================
# CONNECT TO FEDSHIELD FASTAPI BACKEND
# =========================================================

def analyze_url(raw_url):
    """Send the URL to the FastAPI backend and return its result."""

    cleaned_url = raw_url.strip()

    if not cleaned_url:
        return None

    try:
        response = requests.post(
            API_URL,
            json={"url": cleaned_url},
            timeout=30
        )

        if response.status_code != 200:
            try:
                error_detail = response.json().get(
                    "detail",
                    "Unknown backend error"
                )
            except ValueError:
                error_detail = response.text

            return {
                "error": f"Backend error: {error_detail}"
            }

        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "error": (
                "Cannot connect to FedShield backend. "
                "Please start it using: "
                "uvicorn backend:app --reload"
            )
        }

    except requests.exceptions.Timeout:
        return {
            "error": "The analysis timed out. Please try again."
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": f"Request failed: {error}"
        }


def show_authentication():

    st.markdown(
        "<div style='height:45px'></div>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 1.5, 1])

    with center:
        st.markdown(
            "<div class='main-title'>🛡️ FedShield-India</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='subtitle'>Privacy-conscious phishing URL analysis</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='panel'>", unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(
            ["Login", "Create Account"]
        )

        with login_tab:
            st.subheader("Welcome back")

            login_email = st.text_input(
                "Email address",
                key="login_email"
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            if st.button("Login", use_container_width=True):
                if not login_email or not login_password:
                    st.warning("Please enter your email and password.")

                elif verify_user(login_email, login_password):
                    st.session_state.logged_in = True
                    st.session_state.user_id = login_email.strip().lower()
                    st.rerun()

                else:
                    st.error("Incorrect email or password.")

        with register_tab:
            st.subheader("Create your account")

            register_email = st.text_input(
                "Email address",
                key="register_email"
            )

            register_password = st.text_input(
                "Create password",
                type="password",
                key="register_password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password"
            )

            if st.button(
                "Create Account",
                use_container_width=True
            ):
                email = register_email.strip().lower()

                if not valid_email(email):
                    st.error("Please enter a valid email address.")

                elif len(register_password) < 8:
                    st.error("Password must contain at least 8 characters.")

                elif register_password != confirm_password:
                    st.error("Passwords do not match.")

                elif create_user(email, register_password):
                    st.success("Account created. Please login.")

                else:
                    st.error("An account with this email already exists.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Use a separate FedShield password. Never enter your Gmail password."
        )


# =========================================================
# 8. LOGIN CHECK
# =========================================================

if not st.session_state.logged_in:
    show_authentication()
    st.stop()


user_id = st.session_state.user_id


# =========================================================
# 9. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<div style='font-size:25px;font-weight:800;'>🛡️ FedShield</div>",
        unsafe_allow_html=True
    )

    st.caption("INDIA • URL SECURITY PLATFORM")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Check a Link",
            "Scan History",
            "Privacy Center"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("**ACCOUNT**")
    st.caption(user_id)

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.last_result = None
        st.rerun()

    st.markdown("---")

    st.caption("FedShield-India Research Prototype")
    st.caption("Federated Learning • URL Security")


# =========================================================
# 10. HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>FedShield-India</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Understand suspicious links before opening them.</div>",
    unsafe_allow_html=True
)


# =========================================================
# 11. DASHBOARD
# =========================================================

if page == "Dashboard":

    stats = get_user_statistics(user_id)
    history = get_user_history(user_id)

    st.markdown(
        "<div class='section-title'>Security Overview</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "TOTAL SCANS", stats["total"], "Links analyzed"),
        (c2, "HIGH RISK", stats["high_risk"], "Potential threats"),
        (c3, "CAUTION", stats["caution"], "Needs attention"),
        (c4, "LOWER RISK", stats["lower_risk"], "Lower estimated risk")
    ]

    for col, label, value, note in metrics:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>Risk Analytics</div>",
        unsafe_allow_html=True
    )

    if history.empty:
        st.info("No scans yet. Open Check a Link to analyze your first URL.")

    else:
        chart_left, chart_right = st.columns(2)

        risk_counts = pd.DataFrame({
            "Risk Category": [
                "High Risk",
                "Caution",
                "Lower Risk"
            ],
            "Scans": [
                stats["high_risk"],
                stats["caution"],
                stats["lower_risk"]
            ]
        }).set_index("Risk Category")

        with chart_left:
            st.markdown("**Risk Category Breakdown**")
            st.bar_chart(risk_counts)

        with chart_right:
            st.markdown("**Scan Distribution**")
            st.dataframe(
                risk_counts,
                use_container_width=True
            )

        st.markdown(
            "<div class='section-title'>Recent Activity</div>",
            unsafe_allow_html=True
        )

        recent = history.head(5).copy()

        recent["score"] = recent["score"].apply(
            lambda x: f"{float(x):.0f}%"
        )

        recent = recent.rename(columns={
            "url": "URL",
            "risk": "Risk",
            "score": "Phishing Probability",
            "scanned_at": "Date & Time"
        })

        st.dataframe(
            recent[[
                "Date & Time",
                "URL",
                "Risk",
                "Phishing Probability"
            ]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    st.caption(
        "A lower-risk prediction does not guarantee that a website is safe."
    )


# =========================================================
# 12. CHECK A LINK
# =========================================================


elif page == "Check a Link":

    st.markdown(
        "<div class='section-title'>Analyze a URL</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='panel'>"
        "<b>URL Security Scanner</b><br>"
        "<span class='small-muted'>"
        "Enter a complete website address to estimate its phishing risk."
        "</span></div>",
        unsafe_allow_html=True
    )

    # Language selection
    language = st.selectbox(
        "Explanation Language / व्याख्या की भाषा",
        ["English", "Hindi"],
        key="analysis_language"
    )

    is_hindi = language == "Hindi"

    # URL input
    url = st.text_input(
        "वेबसाइट URL" if is_hindi else "Website URL",
        placeholder="https://example.com",
        key="url_input"
    )

    # Analyze button
    if st.button(
        "URL का विश्लेषण करें" if is_hindi else "Analyze URL",
        use_container_width=True
    ):

        cleaned_url = url.strip()

        if not cleaned_url:
            st.warning(
                "कृपया URL दर्ज करें।"
                if is_hindi else
                "Please enter a URL."
            )

        elif not re.match(r"^https?://", cleaned_url, re.IGNORECASE):
            st.warning(
                "कृपया http:// या https:// से शुरू होने वाला URL दर्ज करें।"
                if is_hindi else
                "Enter a URL starting with http:// or https://."
            )

        else:
            with st.spinner(
                "URL की जाँच हो रही है..."
                if is_hindi else
                "Analyzing URL..."
            ):
                result = analyze_url(cleaned_url)

            if result is None:
                st.error("Please enter a URL.")

            elif "error" in result:
                st.error(result["error"])

            else:
                try:
                    save_scan(
                        user_id,
                        cleaned_url,
                        result["risk"],
                        result["score"],
                        result["reasons"]
                    )

                    st.session_state.last_result = result
                    st.session_state.last_url = cleaned_url

                    st.success(
                        "विश्लेषण पूरा हुआ और इतिहास में सुरक्षित है।"
                        if is_hindi else
                        "Analysis completed and saved to your history."
                    )

                except (sqlite3.Error, KeyError, TypeError) as error:
                    st.error(f"Could not save scan: {error}")

    # Display the most recent result
    result = st.session_state.get("last_result")
    last_url = st.session_state.get("last_url", "")

    if result:

        st.markdown("---")

        st.subheader(
            "विश्लेषण परिणाम" if is_hindi else "Analysis Result"
        )

        risk = result.get("risk", "Unknown")
        score = float(result.get("score", 0))
        reasons = result.get("reasons", [])

        if risk == "High Risk":
            st.error(
                "🔴 उच्च जोखिम — संभावित फ़िशिंग लिंक"
                if is_hindi else
                "🔴 HIGH RISK — Potential phishing link"
            )

        elif risk == "Caution":
            st.warning(
                "🟡 सावधानी — इस लिंक की जाँच करें"
                if is_hindi else
                "🟡 CAUTION — Check this link carefully"
            )

        else:
            st.success(
                "🟢 कम अनुमानित जोखिम"
                if is_hindi else
                "🟢 LOWER ESTIMATED RISK"
            )

        st.metric(
            "अनुमानित फ़िशिंग संभावना" if is_hindi
            else "Estimated phishing probability",
            f"{score:.0f}%"
        )

        st.progress(
            min(max(score, 0), 100) / 100
        )

        st.markdown(
            "**जाँचा गया URL**" if is_hindi else "**URL analyzed**"
        )

        st.code(last_url)

        st.markdown(
            "### यह परिणाम क्यों दिखाया गया?"
            if is_hindi else
            "### Why did we show this result?"
        )

        hindi_reasons = {
            "The URL uses HTTP rather than HTTPS.":
                "यह URL HTTPS के बजाय HTTP का उपयोग करता है.",
            "The URL uses an IP address instead of a regular domain name.":
                "यह URL सामान्य डोमेन नाम के बजाय IP एड्रेस का उपयोग करता है.",
            "The URL contains suspicious keywords.":
                "इस URL में संदिग्ध शब्द मौजूद हैं.",
            "The URL is unusually long.":
                "यह URL सामान्य से अधिक लंबा है."
        }

        if reasons:
            for reason in reasons:
                display_reason = (
                    hindi_reasons.get(reason, reason)
                    if is_hindi else reason
                )
                st.write("•", display_reason)
        else:
            st.info(
                "कोई अतिरिक्त कारण उपलब्ध नहीं है।"
                if is_hindi else
                "No additional explanation was provided."
            )

        st.info(
            "सावधानी: यह मशीन लर्निंग मॉडल का अनुमान है। "
            "अपना पासवर्ड या OTP साझा न करें। "
            "संदिग्ध लिंक खोलने से पहले आधिकारिक स्रोत से पुष्टि करें।"
            if is_hindi else
            "Be cautious. This is a machine-learning estimate. "
            "Never share your password or OTP. "
            "Verify suspicious links through official sources."
        )

        st.caption(
            "यह एक शोध प्रोटोटाइप है। कोई भी परिणाम वेबसाइट की सुरक्षा की गारंटी नहीं देता।"
            if is_hindi else
            "This is a research prototype. No result guarantees that a website is safe."
        )


elif page == "Scan History":

    st.markdown(
        "<div class='section-title'>Your Scan History</div>",
        unsafe_allow_html=True
    )

    history = get_user_history(user_id)

    if history.empty:
        st.info("You have not scanned any URLs yet.")

    else:
        st.write(f"Total recorded scans: **{len(history)}**")

        display_history = history.rename(columns={
            "id": "Scan ID",
            "url": "URL",
            "risk": "Risk",
            "score": "Phishing Probability",
            "reasons": "Reasons",
            "scanned_at": "Date & Time"
        })

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Scan History (CSV)",
            data=csv_data,
            file_name="fedshield_scan_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")

        st.warning("Clearing history permanently deletes this account's saved scan records.")

        confirm_clear = st.checkbox(
            "I understand and want to clear my scan history."
        )

        if st.button("Clear My Scan History"):
            if confirm_clear:
                clear_user_history(user_id)
                st.success("Your scan history has been cleared.")
                st.rerun()
            else:
                st.warning("Please confirm before clearing your history.")


# =========================================================
# 14. PRIVACY CENTER
# =========================================================

elif page == "Privacy Center":

    st.markdown(
        "<div class='section-title'>Privacy & Security Center</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3>🔐 Privacy by Design</h3>
        FedShield-India is a research prototype designed to analyze
        suspicious URLs while demonstrating privacy-conscious
        machine learning concepts.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How your data is handled")

    st.write("""
    - Account passwords are stored as salted PBKDF2 hashes.
    - Scan history is stored in a local SQLite database.
    - Scan history is associated with the logged-in email account.
    - URLs are sent to the configured FastAPI backend for analysis.
    - The prototype does not provide email verification.
    """)

    st.markdown("### Federated Learning")

    st.write("""
    FedShield includes a simulated Federated Averaging experiment
    involving multiple clients. The simulation demonstrates how
    model updates can be aggregated without combining the clients'
    original training datasets in one central training dataset.
    """)

    st.warning("""
    Important: This prototype is not a guarantee of website safety
    and does not provide a formal privacy guarantee. Do not use it
    to store sensitive personal information.
    """)

    st.markdown("### System Configuration")

    st.write("Frontend: Streamlit")
    st.write("Backend: FastAPI")
    st.write("Database: SQLite")
    st.write("Machine Learning: URL classification model")
    st.write("Learning approach: Simulated Federated Averaging")

    st.markdown("---")

    st.caption("FedShield-India | Academic Research Prototype")

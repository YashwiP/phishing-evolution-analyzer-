# ============================================================
# Phishing Evolution Analyzer — Streamlit Web App
# Behaviour-Aware Detection of AI-Generated Phishing Emails
#
# HOW TO RUN:
#   1. Upload this file + requirements.txt to your project folder
#   2. pip install -r requirements.txt
#   3. streamlit run app.py
#
# OR deploy free at: https://streamlit.io/cloud
# ============================================================

import re
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Evolution Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #58a6ff !important;
}

/* Cards */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

.metric-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

/* Verdict badges */
.verdict-legit {
    background: linear-gradient(135deg, #1a3a1a, #1f4a1f);
    border: 2px solid #3fb950;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #3fb950;
    font-weight: 700;
}

.verdict-trad {
    background: linear-gradient(135deg, #3a1a1a, #4a1f1f);
    border: 2px solid #f85149;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #f85149;
    font-weight: 700;
}

.verdict-ai {
    background: linear-gradient(135deg, #2a1a3a, #351f4a);
    border: 2px solid #d2a8ff;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #d2a8ff;
    font-weight: 700;
}

.bmi-bar-container {
    background: #21262d;
    border-radius: 20px;
    height: 18px;
    width: 100%;
    margin: 8px 0;
    overflow: hidden;
}

.feature-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 3px;
}

.pill-green  { background:#1a3a1a; border:1px solid #3fb950; color:#3fb950; }
.pill-red    { background:#3a1a1a; border:1px solid #f85149; color:#f85149; }
.pill-yellow { background:#3a2e1a; border:1px solid #d29922; color:#d29922; }
.pill-purple { background:#2a1a3a; border:1px solid #d2a8ff; color:#d2a8ff; }

.info-banner {
    background: #0c2844;
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #cdd9e5;
}

stTextArea textarea {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION ENGINE
# (Lightweight version — no LanguageTool/spaCy dependency
#  so the app runs instantly without heavy installs)
# ═══════════════════════════════════════════════════════════════

# ── Keyword lists ─────────────────────────────────────────────
URGENCY_KEYWORDS = [
"immediately", "urgent", "asap", "right away", "act now",
    "within 24 hours", "within 48 hours", "within 12 hours",
    "within 6 hours", "within 7 days",
    "expire", "expires", "will expire", "will expire in",
    "expire in 24", "expire in 48", "expiring soon",
    "final notice", "warning", "alert", "critical",
    "failure to", "will be deleted", "will be suspended",
    "will be terminated", "will be cancelled", "will be closed",
    "last chance", "limited time", "do not delay", "do not ignore",
    "password will expire", "account will expire",
    "update your password", "reset your password",
    "verify your identity", "confirm your identity",
    "update immediately", "respond immediately",
    "click the link below", "follow the link below",
    "click here immediately", "take action now"
]

AUTHORITY_KEYWORDS = [
    "security team", "it department", "hr department", "support team",
    "compliance", "official", "authorized", "verified", "certified",
    "management", "director", "executive", "bank", "government",
    "federal", "internal revenue", "identity verification",
    "security operations", "account security", "fraud prevention",
    "risk management", "headquarters"
]

GRAMMAR_ERROR_PATTERNS = [
    r"\byou\s+account\b", r"\byou\s+information\b", r"\byou\s+password\b",
    r"\bwe\s+was\b", r"\bthey\s+was\b", r"\bi\s+has\b",
    r"\bhas\s+been\s+detect\b", r"\bhas\s+been\s+suspend\b",
    r"\bwill\s+be\s+cancel\b", r"\bwill\s+be\s+delete\b",
    r"\bdo\s+not\s+ignor\b", r"\bplease\s+login\s+to\s+your\s+acount\b",
    r"\bwe\s+have\s+notice\b", r"\bwe\s+was\s+unable\b",
]

COMMON_MISSPELLINGS = [
    "activty", "temporarly", "informations", "acces", "permanant",
    "selectd", "anual", "recieve", "adress", "immidiately",
    "permenantly", "unknow", "procces", "trasaction", "recipent",
    "imediatley", "cancell", "ignor", "mesage", "loose",
    "expirez", "oppertunity", "forfieture", "sevice", "securty",
    "departmant", "cliam", "approvd", "attemps", "unlocked",
    "personel", "eligble"
]

GREETING_NAMED    = re.compile(r"^(hi|hello|dear|hey)\s+[A-Z][a-z]+", re.I)
GREETING_GENERIC  = re.compile(
    r"^(dear\s+(valued|customer|user|member|sir|madam|winner|all|team"
    r"|network\s+user|account\s+holder|subscriber|colleague|employee)"
    r"|hello\s+dear|to whom|dear\s+taxpayer|apple\s+user"
    r"|dear\s+all|attention\s+user|greetings\s+user)", re.I)
CLOSING_PATTERNS  = re.compile(
    r"^(regards|best|sincerely|cheers|thanks|thank you|warm regards|"
    r"kind regards|yours|respectfully|cordially|best wishes)", re.I)
URL_PATTERN       = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.I)
SUSPICIOUS_DOMAIN = re.compile(
    r"https?://[^\s]*(\.xyz|\.club|\.net/[a-z\-]+verify|"
    r"\.net/[a-z\-]+login|[a-z\-]+-secure[a-z\-]*\.|"
    r"[a-z\-]+-verify[a-z\-]*\.|[a-z\-]+-alert[a-z\-]*\.)", re.I)


# ── Utility helpers ───────────────────────────────────────────
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_structure(body):
    lines     = [l.strip() for l in body.split("\n") if l.strip()]
    greeting  = ""
    signature = ""
    start_idx = 0

    if not lines:
        return {"greeting": "", "core_body": "", "signature": ""}

    for i, line in enumerate(lines[:2]):
        if GREETING_NAMED.match(line) or GREETING_GENERIC.match(line):
            greeting  = line
            start_idx = i + 1
            break

    end_idx        = len(lines)
    closing_found  = False
    for i in range(len(lines)-1, start_idx-1, -1):
        if CLOSING_PATTERNS.match(lines[i]) and not closing_found:
            closing_found = True
            end_idx       = i
        elif closing_found and i >= end_idx - 3:
            end_idx = i
        elif closing_found:
            break

    if closing_found:
        signature = " | ".join(lines[end_idx:])
    else:
        signature = " | ".join(lines[-2:])
        end_idx   = len(lines) - 2

    core_body = " ".join(lines[start_idx:end_idx])
    return {"greeting": greeting, "core_body": core_body, "signature": signature}


# ── Feature functions ─────────────────────────────────────────
def count_grammar_errors_fast(text):
    t = text.lower()
    return sum(1 for p in GRAMMAR_ERROR_PATTERNS if re.search(p, t))


def count_spelling_errors_fast(text):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return sum(1 for w in words if w in COMMON_MISSPELLINGS)


def count_urls(text):
    return len(URL_PATTERN.findall(text))


def count_suspicious_urls(text):
    return len(SUSPICIOUS_DOMAIN.findall(text))


def urgency_score(text):
    t    = text.lower()
    hits = sum(1 for kw in URGENCY_KEYWORDS if kw in t)
    return round(min(hits / 5.0, 1.0), 2)


def authority_score(text):
    t    = text.lower()
    hits = sum(1 for kw in AUTHORITY_KEYWORDS if kw in t)
    return round(min(hits / 4.0, 1.0), 2)


def personalization_score(greeting, core_body):
    score = 0.0
    text  = greeting + " " + core_body
    if GREETING_NAMED.match(greeting):
        score += 0.4
    if re.search(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", core_body):
        score += 0.3
    if re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\$[\d,]+|#[A-Z0-9\-]+|\d{4,})\b", text):
        score += 0.3
    return round(min(score, 1.0), 2)


def greeting_realism(greeting):
    if not greeting:
        return 0.0
    if GREETING_NAMED.match(greeting):
        return 1.0
    if GREETING_GENERIC.match(greeting):
        return 0.3
    return 0.5


def signature_realism(signature):
    if not signature:
        return 0.0
    score     = 0.0
    sig_lower = signature.lower()
    if re.search(r"[A-Z][a-z]+\s[A-Z][a-z]+", signature):
        score += 0.3
    title_words = ["manager","director","team","department","executive",
                   "officer","analyst","specialist","advisor","lead","head"]
    if any(w in sig_lower for w in title_words):
        score += 0.3
    org_words = ["ltd","inc","corp","group","services","solutions",
                 "bank","university","institute","agency"]
    if any(w in sig_lower for w in org_words):
        score += 0.4
    return round(min(score, 1.0), 2)


def lexical_diversity(text):
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    return round(np.mean(lengths), 2)


# ── BMI Computation ───────────────────────────────────────────
BMI_WEIGHTS = {
    "personalization" : 0.20,
    "greeting_realism": 0.15,
    "sig_realism"     : 0.15,
    "authority_score" : 0.10,
    "urgency_score"   : 0.05,
    "grammar_errors"  : 0.20,
    "spelling_errors" : 0.15,
}

def compute_bmi(features):
    g_norm = min(features["grammar_errors"]  / 5.0, 1.0)
    s_norm = min(features["spelling_errors"] / 5.0, 1.0)
    bmi = (
        features["personalization"]  * BMI_WEIGHTS["personalization"]  +
        features["greeting_realism"] * BMI_WEIGHTS["greeting_realism"] +
        features["sig_realism"]      * BMI_WEIGHTS["sig_realism"]      +
        features["authority_score"]  * BMI_WEIGHTS["authority_score"]  +
        features["urgency_score"]    * BMI_WEIGHTS["urgency_score"]    +
        (1 - g_norm)                 * BMI_WEIGHTS["grammar_errors"]   +
        (1 - s_norm)                 * BMI_WEIGHTS["spelling_errors"]
    )
    return round(float(bmi), 4)


# ── Rule-based classifier ─────────────────────────────────────
def classify_email(features, bmi):
    g               = features["grammar_errors"]
    s               = features["spelling_errors"]
    u               = features["urgency_score"]
    p               = features["personalization"]
    gr              = features["greeting_realism"]
    sr              = features["sig_realism"]
    suspicious_urls = features["suspicious_urls"]
    urls            = features["url_count"]

    # ── Rule 1: Traditional Phishing ──────────────────────────
    # High errors OR very generic with urgency
    if (g >= 3 or s >= 3) and u >= 0.2:
        return 1, "Traditional Phishing"

    # Generic greeting + urgency + no signature = traditional
    if gr <= 0.3 and u >= 0.2 and sr < 0.3:
        return 1, "Traditional Phishing"

    # ── Rule 2: AI Phishing ───────────────────────────────────
    # Low errors + any urgency + URL present = AI phishing
    if g <= 1 and s <= 1 and u >= 0.2 and urls >= 1:
        return 2, "AI-Generated Phishing"

    # Low errors + urgency even without URL = suspicious
    if g <= 1 and s <= 1 and u >= 0.3:
        return 2, "AI-Generated Phishing"

    # Suspicious URL regardless of other signals
    if suspicious_urls > 0:
        return 2, "AI-Generated Phishing"

    # ── Rule 3: Legitimate ────────────────────────────────────
    # Only legitimate if: no errors, low urgency, well-signed
    if g == 0 and s == 0 and u <= 0.15 and bmi >= 0.65:
        return 0, "Legitimate"

    # ── Fallback: BMI thresholds ──────────────────────────────
    if u >= 0.2:
        # Any urgency pushes toward phishing
        if g <= 1 and s <= 1:
            return 2, "AI-Generated Phishing"
        else:
            return 1, "Traditional Phishing"

    if bmi >= 0.70:
        return 0, "Legitimate"
    elif bmi >= 0.40:
        return 2, "AI-Generated Phishing"
    else:
        return 1, "Traditional Phishing"


# ── Full analysis pipeline ────────────────────────────────────
def analyze_email(subject, body):
    full_text = subject + " " + body
    cleaned   = clean_text(full_text)
    structure = extract_structure(clean_text(body))

    features = {
        "grammar_errors"  : count_grammar_errors_fast(cleaned),
        "spelling_errors" : count_spelling_errors_fast(cleaned),
        "url_count"       : count_urls(cleaned),
        "suspicious_urls" : count_suspicious_urls(cleaned),
        "urgency_score"   : urgency_score(cleaned),
        "authority_score" : authority_score(cleaned),
        "personalization" : personalization_score(
                                structure["greeting"],
                                structure["core_body"]),
        "greeting_realism": greeting_realism(structure["greeting"]),
        "sig_realism"     : signature_realism(structure["signature"]),
        "lex_diversity"   : lexical_diversity(cleaned),
        "avg_sent_len"    : avg_sentence_length(cleaned),
    }

    bmi                   = compute_bmi(features)
    label, label_name     = classify_email(features, bmi)
    features["BMI"]       = bmi
    features["label"]     = label
    features["label_name"]= label_name
    features["greeting"]  = structure["greeting"]
    features["signature"] = structure["signature"]
    return features


# ═══════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Phishing Analyzer")
    st.markdown("*Behaviour-Aware Detection*")
    st.markdown("---")
    st.markdown("""
**Research Project**
IEEE-style paper supporting tool.

**3 Email Classes Detected:**
- ✅ Legitimate
- 🚨 Traditional Phishing
- 🤖 AI-Generated Phishing

**Novel Contribution:**
Behavioural Mimicry Index (BMI)
measures how well a phishing email
mimics legitimate behaviour — beyond
grammar checking alone.
    """)
    st.markdown("---")
    st.markdown("**Feature Categories:**")
    st.markdown("🔤 Linguistic (grammar, spelling)")
    st.markdown("🔗 Semantic (URLs, entities)")
    st.markdown("🧠 Behavioural (BMI components)")
    st.markdown("---")
    st.markdown(
        "<small>Built with Python · spaCy · scikit-learn<br>"
        "Research by: Phishing Evolution Analyzer</small>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
# UI — MAIN
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🛡️ Phishing Evolution Analyzer")
st.markdown(
    "##### Behaviour-Aware Detection of AI-Generated Phishing Emails"
)
st.markdown(
    '<div class="info-banner">🔬 <b>Research Tool</b> — '
    'This system detects not just traditional phishing (bad grammar, generic greetings) '
    'but also <b>AI-generated phishing</b> that is perfectly written and personalized. '
    'It uses a novel <b>Behavioural Mimicry Index (BMI)</b> to score how well an email '
    'mimics legitimate behaviour.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ── Input section ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📧 Email Input")
    subject_input = st.text_input(
        "Subject Line",
        placeholder="e.g. URGENT: Your account has been suspended",
        help="Paste the email subject line here"
    )
    body_input = st.text_area(
        "Email Body",
        height=280,
        placeholder=(
            "Paste the full email body here...\n\n"
            "Include the greeting, main content, and signature "
            "for best results."
        ),
        help="Paste the complete email body including greeting and signature"
    )

    analyze_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Try a sample:")
    col_s1, col_s2, col_s3 = st.columns(3)

    SAMPLE_LEGIT = {
        "subject": "Q3 Project Update — Action Required",
        "body": (
            "Hi Sarah,\n\n"
            "I hope you're doing well. I wanted to touch base regarding the Q3 "
            "project deliverables. As discussed in last Tuesday's meeting, the "
            "deadline is set for October 15th.\n\n"
            "Could you please review the attached report and share your feedback "
            "by end of week? Let me know if you have any questions.\n\n"
            "Best regards,\nMichael Thompson\nProject Manager, Acme Corp"
        )
    }

    SAMPLE_TRAD = {
        "subject": "URGENT: Your Account Has Been Suspended!!!",
        "body": (
            "Dear Valued Customer,\n\n"
            "We have detected unusual activty on you account. Your account has "
            "been temporarly suspended for you safety.\n\n"
            "You must verify you informations IMMEDIATELY or your account will "
            "be DELETED. Click the link below to restore acces:\n\n"
            "http://secure-login-verify.xyz/update-account\n\n"
            "Failure to act within 24 hours will result in permanant suspension.\n\n"
            "Regards,\nThe Security Team"
        )
    }

    SAMPLE_AI = {
        "subject": "Action Required: Unusual Sign-In Detected on Your Account",
        "body": (
            "Hi Sarah,\n\n"
            "We noticed a sign-in to your account from a device we don't recognize. "
            "The sign-in occurred on October 8th at 11:42 PM from Chicago, IL.\n\n"
            "If this was you, no action is needed. However, if you don't recognize "
            "this activity, we recommend securing your account immediately:\n\n"
            "https://accounts.secure-id-verify.com/review-activity\n\n"
            "For your protection, this link will expire in 24 hours.\n\n"
            "Best regards,\nThe Account Security Team\nTrustBank Online Services"
        )
    }

    if col_s1.button("✅ Legitimate", use_container_width=True):
        st.session_state["sample"] = SAMPLE_LEGIT
    if col_s2.button("🚨 Traditional", use_container_width=True):
        st.session_state["sample"] = SAMPLE_TRAD
    if col_s3.button("🤖 AI Phishing", use_container_width=True):
        st.session_state["sample"] = SAMPLE_AI

    if "sample" in st.session_state:
        s = st.session_state["sample"]
        st.info(
            f"**Sample loaded** — copy the text above and paste it into "
            f"the input fields, or re-run analysis.\n\n"
            f"**Subject:** {s['subject']}\n\n"
            f"**Body preview:** {s['body'][:120]}..."
        )

# ── Results section ───────────────────────────────────────────
with col_right:
    st.markdown("### 📊 Analysis Results")

    # Auto-fill from sample
    if "sample" in st.session_state and not analyze_btn:
        s              = st.session_state["sample"]
        subject_input  = s["subject"]
        body_input     = s["body"]

    if analyze_btn or "sample" in st.session_state:
        if not body_input or len(body_input.strip()) < 20:
            st.warning("⚠️ Please paste an email body (at least a few sentences).")
        else:
            with st.spinner("Analyzing email..."):
                result = analyze_email(subject_input or "", body_input)

            label      = result["label"]
            label_name = result["label_name"]
            bmi        = result["BMI"]

            # ── Verdict badge ──────────────────────────────────
            if label == 0:
                verdict_class = "verdict-legit"
                verdict_icon  = "✅"
                verdict_msg   = "LEGITIMATE EMAIL"
                verdict_desc  = "This email shows normal behavioural patterns."
            elif label == 1:
                verdict_class = "verdict-trad"
                verdict_icon  = "🚨"
                verdict_msg   = "TRADITIONAL PHISHING"
                verdict_desc  = "Grammar errors and urgency tactics detected."
            else:
                verdict_class = "verdict-ai"
                verdict_icon  = "🤖"
                verdict_msg   = "AI-GENERATED PHISHING"
                verdict_desc  = "Polished writing but suspicious behavioural signals."

            st.markdown(
                f'<div class="{verdict_class}">'
                f'{verdict_icon} {verdict_msg}<br>'
                f'<small style="font-family:DM Sans;font-weight:300;font-size:0.85rem;">'
                f'{verdict_desc}</small>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # ── BMI Score ──────────────────────────────────────
            st.markdown("#### 🧠 Behavioural Mimicry Index (BMI)")
            bmi_pct   = int(bmi * 100)
            bmi_color = "#3fb950" if label==0 else ("#d2a8ff" if label==2 else "#f85149")

            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:4px;">'
                f'<span style="font-family:Space Mono;font-size:0.8rem;color:#8b949e;">'
                f'Low Mimicry (Obvious)</span>'
                f'<span style="font-family:Space Mono;font-size:1.3rem;'
                f'color:{bmi_color};font-weight:700;">{bmi:.3f}</span>'
                f'<span style="font-family:Space Mono;font-size:0.8rem;color:#8b949e;">'
                f'High Mimicry (Dangerous)</span>'
                f'</div>'
                f'<div class="bmi-bar-container">'
                f'<div style="height:100%;width:{bmi_pct}%;'
                f'background:linear-gradient(90deg,#1f6feb,{bmi_color});'
                f'border-radius:20px;transition:width 0.5s;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # ── Feature breakdown ──────────────────────────────
            st.markdown("#### 📋 Feature Breakdown")

            m1, m2, m3 = st.columns(3)
            m1.metric("Grammar Errors",  result["grammar_errors"])
            m2.metric("Spelling Errors", result["spelling_errors"])
            m3.metric("URLs Found",      result["url_count"])

            m4, m5, m6 = st.columns(3)
            m4.metric("Urgency Score",    f"{result['urgency_score']:.2f}")
            m5.metric("Authority Score",  f"{result['authority_score']:.2f}")
            m6.metric("Personalization",  f"{result['personalization']:.2f}")

            m7, m8, m9 = st.columns(3)
            m7.metric("Greeting Realism", f"{result['greeting_realism']:.2f}")
            m8.metric("Sig. Realism",     f"{result['sig_realism']:.2f}")
            m9.metric("Lex. Diversity",   f"{result['lex_diversity']:.3f}")

            st.markdown("---")

            # ── Signal pills ───────────────────────────────────
            st.markdown("#### 🏷️ Detected Signals")
            pills_html = ""

            if result["grammar_errors"] > 0:
                pills_html += f'<span class="feature-pill pill-red">⚠️ Grammar errors: {result["grammar_errors"]}</span>'
            else:
                pills_html += '<span class="feature-pill pill-green">✅ No grammar errors</span>'

            if result["spelling_errors"] > 0:
                pills_html += f'<span class="feature-pill pill-red">⚠️ Spelling errors: {result["spelling_errors"]}</span>'
            else:
                pills_html += '<span class="feature-pill pill-green">✅ No spelling errors</span>'

            if result["urgency_score"] > 0.4:
                pills_html += '<span class="feature-pill pill-red">🚨 High urgency language</span>'
            elif result["urgency_score"] > 0.1:
                pills_html += '<span class="feature-pill pill-yellow">⚡ Mild urgency language</span>'

            if result["suspicious_urls"] > 0:
                pills_html += f'<span class="feature-pill pill-red">🔗 Suspicious URL detected</span>'
            elif result["url_count"] > 0:
                pills_html += f'<span class="feature-pill pill-yellow">🔗 URL present</span>'

            if result["personalization"] >= 0.7:
                pills_html += '<span class="feature-pill pill-purple">🤖 Highly personalized</span>'
            elif result["personalization"] < 0.3:
                pills_html += '<span class="feature-pill pill-yellow">👤 Generic / impersonal</span>'

            if result["greeting_realism"] == 1.0:
                pills_html += '<span class="feature-pill pill-purple">👋 Named greeting</span>'
            elif result["greeting_realism"] <= 0.3:
                pills_html += '<span class="feature-pill pill-yellow">👋 Generic greeting</span>'

            if result["sig_realism"] >= 0.7:
                pills_html += '<span class="feature-pill pill-purple">✍️ Professional signature</span>'
            else:
                pills_html += '<span class="feature-pill pill-yellow">✍️ Weak signature</span>'

            if result["authority_score"] >= 0.5:
                pills_html += '<span class="feature-pill pill-purple">🏛️ Authority language</span>'

            st.markdown(pills_html, unsafe_allow_html=True)

            # ── Extracted structure ────────────────────────────
            with st.expander("🔍 View Extracted Email Structure"):
                st.markdown(f"**Greeting detected:** `{result['greeting'] or '(none)'}`")
                st.markdown(f"**Signature detected:** `{result['signature'] or '(none)'}`")

# ═══════════════════════════════════════════════════════════════
# UI — ABOUT / RESEARCH SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📖 About This Research")

tab1, tab2, tab3 = st.tabs([
    "🎯 Research Problem",
    "🧮 BMI Formula",
    "📊 Model Architecture"
])

with tab1:
    st.markdown("""
**The Evolution of Phishing Attacks**

Traditional phishing emails were easy to detect — they contained obvious grammar
mistakes, spelling errors, generic greetings like *"Dear Valued Customer"*, and
urgent threats. Simple rule-based or ML classifiers trained on linguistic features
could catch them reliably.

**The New Threat: AI-Generated Phishing**

Modern attackers use AI tools to generate phishing emails that are:
- ✅ Grammatically perfect
- ✅ Professionally signed
- ✅ Personalized with real names and details
- ✅ Contextually believable (fake invoice, account alert, etc.)

These emails **bypass traditional grammar-based detectors entirely**.

**Our Solution**

We introduce a *Behaviour-Aware Detection* system that looks at HOW an email
behaves — not just HOW it's written. The novel **Behavioural Mimicry Index (BMI)**
quantifies how well a phishing email mimics legitimate communication.
    """)

with tab2:
    st.markdown("""
**Behavioural Mimicry Index Formula**

```
BMI = (Personalization      × 0.20)
    + (Greeting Realism      × 0.15)
    + (Signature Realism     × 0.15)
    + (Authority Score       × 0.10)
    + (Urgency Score         × 0.05)
    + ((1 - Grammar Errors)  × 0.20)
    + ((1 - Spelling Errors) × 0.15)
```

**BMI Interpretation:**

| BMI Range | Interpretation |
|-----------|---------------|
| 0.70 – 1.00 | Legitimate or very convincing AI phishing |
| 0.45 – 0.69 | Possibly AI-generated phishing |
| 0.00 – 0.44 | Traditional phishing (poor mimicry) |

**Key Insight:** AI phishing achieves BMI scores close to legitimate emails,
which is exactly why grammar-only models fail to detect them.
    """)

with tab3:
    st.markdown("""
**Three Models — Three Perspectives**

| Model | Algorithm | Features | Research Role |
|-------|-----------|----------|---------------|
| **Model A** | Logistic Regression | Linguistic only | Baseline — fails on AI phishing |
| **Model B** | Random Forest | Behavioural only | Improves AI phishing recall |
| **Model C** | Gradient Boosting | Combined + BMI | Full proposed system |

**The Core Finding:**

Model A (grammar-only) has strong recall for Traditional Phishing but
**poor recall for AI Phishing** — proving the limitation of existing systems.

Model C (with BMI) achieves measurably improved recall for AI Phishing —
proving that behavioural features are necessary for modern phishing detection.
    """)

st.markdown("---")
st.markdown(
    "<center><small style='color:#8b949e;font-family:Space Mono,monospace;'>"
    "Phishing Evolution Analyzer · IEEE Research Project · "
    "Built with Python, spaCy, scikit-learn, Streamlit"
    "</small></center>",
    unsafe_allow_html=True
)

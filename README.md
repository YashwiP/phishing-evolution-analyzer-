# Phishing Evolution Analyzer
### Behaviour-Aware Detection of AI-Generated Phishing Emails

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat&logo=spacy&logoColor=white)
![Research](https://img.shields.io/badge/IEEE--Style-Research-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Overview

Phishing emails have evolved. Traditional spam filters were built to catch emails with **bad grammar and spelling mistakes**. But modern attackers now use AI tools to generate phishing emails that are:

- Grammatically perfect
- Professionally written and signed
- Personalized with real names and contextual details
- Indistinguishable from legitimate emails — to grammar-based detectors

This project builds a **behaviour-aware detection system** that goes beyond grammar checking. It introduces a novel metric — the **Behavioural Mimicry Index (BMI)** — that measures how well a phishing email mimics legitimate communication behaviour.

> **Core Research Finding:** Grammar-based models achieve strong recall on traditional phishing but fail significantly on AI-generated phishing. Behavioural features + BMI measurably improve AI phishing detection.

---

## Project Goals

| Goal | Description |
|------|-------------|
| **Dataset** | 3-class labeled dataset: Legitimate, Traditional Phishing, AI Phishing |
| **Feature Extraction** | 13 features across Linguistic, Semantic, and Behavioural categories |
| **Novel Metric** | Behavioural Mimicry Index (BMI) — weighted combination of behavioural signals |
| **Model Comparison** | 3 models trained on different feature sets to prove the research hypothesis |
| **Web App** | Interactive Streamlit app for real-time email analysis |

---

## The Behavioural Mimicry Index (BMI)

The BMI is the **novel contribution** of this research. It is a single score (0.0 – 1.0) that quantifies how convincingly an email mimics legitimate behaviour.

```
BMI = (Personalization Score   × 0.20)
    + (Greeting Realism        × 0.15)
    + (Signature Realism       × 0.15)
    + (Authority Score         × 0.10)
    + (Urgency Score           × 0.05)
    + ((1 - Grammar Errors)    × 0.20)
    + ((1 - Spelling Errors)   × 0.15)
```

| BMI Range | Interpretation |
|-----------|---------------|
| 0.70 – 1.00 | Legitimate or highly convincing AI phishing |
| 0.45 – 0.69 | Potentially AI-generated phishing |
| 0.00 – 0.44 | Traditional phishing (poor mimicry) |

---

## 📊 Feature Categories

### A) Linguistic Features
| Feature | Description |
|---------|-------------|
| `grammar_errors` | Count of grammar mistakes via LanguageTool |
| `spelling_errors` | Count of misspelled words via pyspellchecker |
| `flesch_score` | Flesch Reading Ease readability score |
| `avg_sent_len` | Average words per sentence |
| `lex_diversity` | Type-Token Ratio (unique / total words) |

### B) Semantic & Contextual Features
| Feature | Description |
|---------|-------------|
| `ner_count` | Named entity count via spaCy NER |
| `url_count` | Number of URLs in the email |
| `ctx_richness` | Ratio of content words to all words |

### C) Behavioural Features
| Feature | Description |
|---------|-------------|
| `personalization` | Score for named greetings, specific details |
| `greeting_realism` | Named vs generic salutation score |
| `sig_realism` | Completeness of name/title/organisation signature |
| `urgency_score` | Normalised count of urgency language |
| `authority_score` | Normalised count of authority/trust language |

---

## 🤖 Models Trained

| Model | Algorithm | Feature Set | Research Role |
|-------|-----------|-------------|---------------|
| **Model A** | Logistic Regression | Linguistic only (5 features) | Baseline — proves grammar model fails on AI phishing |
| **Model B** | Random Forest | Behavioural only (8 features) | Shows behavioural features improve AI phishing recall |
| **Model C** | Gradient Boosting | All features + BMI (14 features) | Full proposed system — best overall performance |

---

## Project Structure

```
phishing-evolution-analyzer/
│
├── 📓 Phishing_Evolution_Analyzer.ipynb   ← Full Google Colab notebook
├── 🌐 app.py                              ← Streamlit web application
├── 📋 requirements.txt                    ← Python dependencies
│
├── data/
│   ├── raw/
│   │   └── email_dataset.csv              ← 30-email labeled dataset
│   └── processed/
│       └── email_dataset_processed.csv    ← Cleaned + structured emails
│
├── features/
│   └── feature_matrix.csv                 ← 13 features + BMI per email
│
├── models/
│   ├── model_A_logistic_regression.pkl    ← Trained baseline model
│   ├── model_B_random_forest.pkl          ← Trained behavioural model
│   └── model_C_gradient_boosting.pkl      ← Trained full model
│
└── outputs/
    ├── plots/
    │   ├── class_distribution.png
    │   ├── feature_heatmap.png
    │   ├── bmi_distribution.png
    │   ├── feature_importance.png
    │   ├── confusion_matrices.png
    │   ├── recall_comparison.png
    │   ├── model_summary.png
    │   └── research_dashboard.png         ← ⭐ Main results figure
    └── reports/
        ├── model_comparison.csv
        └── full_results_report.csv
```

---

## How to Run

### Option 1 — Google Colab (Recommended for full pipeline)
1. Open `Phishing_Evolution_Analyzer.ipynb` in Google Colab
2. Run all cells from top to bottom
3. All outputs are saved automatically

### Option 2 — Streamlit Web App (Live demo)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/phishing-evolution-analyzer.git
cd phishing-evolution-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option 3 — Deploy Free on Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click Deploy — live URL generated instantly

---

## Technical Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Google Colab** | Research environment |
| **spaCy** | NLP — NER, tokenization, POS tagging |
| **LanguageTool** | Grammar error detection |
| **pyspellchecker** | Spelling error detection |
| **textstat** | Readability scores (Flesch) |
| **scikit-learn** | ML models, evaluation metrics |
| **pandas / numpy** | Data handling |
| **matplotlib / seaborn** | Visualisation |
| **Streamlit** | Web application |

---

## Key Results

> Results are based on the research prototype dataset (30 emails).
> The pattern of findings — not absolute numbers — is the research contribution.

**AI Phishing Recall Comparison:**

| Model | Traditional Phishing Recall | AI Phishing Recall |
|-------|-----------------------------|--------------------|
| Model A (Grammar-based) | High | **Low** ← the problem |
| Model B (Behavioural) | Moderate | **Improved** |
| Model C (Combined + BMI) | High | **Best** ← the solution |

**Key Finding:** The grammar-only baseline achieves strong recall on traditional
phishing but fails on AI-generated phishing. The proposed behaviour-aware system
with BMI measurably improves AI phishing recall — validating the research hypothesis.

---

## Research Context

This project supports an **IEEE-style research paper** titled:

> *"Phishing Evolution Analyzer: Behaviour-Aware Detection of AI-Generated Phishing Emails"*

**Abstract Summary:**
As AI tools become accessible to malicious actors, phishing emails have evolved
from grammar-error-riddled scams to polished, behavioural mimicry attacks.
This paper proposes a behaviour-aware detection framework introducing the
Behavioural Mimicry Index (BMI) — a novel metric that captures how convincingly
a phishing email mimics legitimate communication. Experimental results demonstrate
that grammar-based models fail on AI-generated phishing while the proposed
combined system achieves improved detection across all three email classes.

---

## Author

**Yashwi Pandey**
- pandeyyashwii@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/yashwi-pandey/)

---

## License

This project is licensed under the MIT License.
Free to use for research and educational purposes with attribution.

---

## If You Found This Useful

Give this repository a ⭐ — it helps other researchers find the project.

---

<p align="center">
  Built with Python · spaCy · scikit-learn · Streamlit<br>
  <i>Phishing Evolution Analyzer — IEEE Research Project</i>
</p>

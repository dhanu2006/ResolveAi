"""
nlp_engine.py
Core NLP pipeline for Resolve.AI:
  - Complaint classification (category + severity)
  - Sentiment analysis
  - Duplicate / similarity detection
  - Gen-AI response drafting via HuggingFace Inference API
"""
import os
import re
import json
import requests
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import pandas as pd

# ---------------------------------------------------------------------------
# Training data — keyword-anchored examples per category
# ---------------------------------------------------------------------------
TRAINING_DATA = {
    "ATM / Debit Card": [
        "atm card swallowed cash not dispensed",
        "debit card not working pos terminal declined",
        "atm transaction failed amount deducted",
        "card blocked without reason unable to use",
        "cash withdrawal failed money debited atm",
        "debit card declined merchant shop",
        "atm machine not giving money debited account",
        "card expired new card not received",
    ],
    "Net Banking / Mobile App": [
        "unable login mobile banking app invalid credentials",
        "fund transfer failed amount deducted net banking",
        "net banking session timeout transaction incomplete",
        "otp not received unable register mobile banking",
        "upi payment failed money debited not refunded",
        "internet banking password reset problem",
        "transaction history wrong entries net banking",
        "mobile app crashing unable access account",
    ],
    "Loan / EMI": [
        "emi debited twice double deduction loan account",
        "loan statement incorrect outstanding amount",
        "personal loan application no response pending",
        "prepayment charges wrongly applied floating rate",
        "cibil score wrongly updated loan default",
        "home loan sanction letter not received",
        "emi bounce charges levied incorrect",
        "loan closure certificate not issued",
    ],
    "Account / KYC": [
        "account frozen blocked without notice",
        "kyc documents submitted account still restricted",
        "wrong name spelling passbook debit card",
        "unable update mobile number bank account",
        "account opened without consent fraud",
        "joint account holder removed unauthorized",
        "address change request not processed",
        "savings account minimum balance charges unfair",
    ],
    "Fraud / Unauthorized Transaction": [
        "unauthorized transaction money debited fraud",
        "phishing call account debited fraudulently",
        "otp shared unknowingly money transferred stolen",
        "credit card details stolen online shopping",
        "multiple transactions not authorized suspicious",
        "fraudulent fd opened without knowledge",
        "money gone from account hack cyber fraud",
        "sim swap fraud account compromised",
    ],
    "Branch Service": [
        "bank staff rude unhelpful branch poor service",
        "cheque deposited not cleared delayed",
        "passbook update machine not working branch",
        "dd demand draft taking too long",
        "neft submitted branch money not received beneficiary",
        "long waiting queue branch understaffed",
        "branch manager not available complaint ignored",
        "locker facility issue branch unresponsive",
    ],
}

SEVERITY_KEYWORDS = {
    "Critical": [
        "fraud", "stolen", "hacked", "unauthorized", "phishing", "otp shared",
        "sim swap", "cyber crime", "45000", "85000", "lakh",
    ],
    "High": [
        "urgent", "immediately", "deducted", "debited", "failed", "blocked",
        "frozen", "loan", "emi", "double", "twice",
    ],
    "Medium": [
        "not working", "unable", "incorrect", "wrong", "pending", "delayed",
        "not received", "not processed",
    ],
    "Low": [
        "rude", "waiting", "slow", "queue", "passbook", "address", "name",
    ],
}

SENTIMENT_KEYWORDS = {
    "Angry": [
        "disgusting", "worst", "fraud", "cheated", "stolen", "immediately",
        "unacceptable", "irresponsible", "incompetent", "useless",
    ],
    "Frustrated": [
        "again", "still", "repeatedly", "multiple times", "no response",
        "ignored", "waiting", "days", "weeks", "not resolved",
    ],
    "Concerned": [
        "worried", "concern", "please help", "request", "kindly",
        "confused", "not sure", "don't understand",
    ],
    "Neutral": [],
}


class ResolveAINLPEngine:
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.all_texts = []
        self.all_categories = []
        self._build_training_corpus()
        self._train_classifier()

    def _build_training_corpus(self):
        for category, texts in TRAINING_DATA.items():
            for text in texts:
                self.all_texts.append(text)
                self.all_categories.append(category)

    def _train_classifier(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=500,
                C=1.0,
                random_state=42,
            )),
        ])
        self.pipeline.fit(self.all_texts, self.all_categories)
        self.vectorizer = self.pipeline.named_steps["tfidf"]

    def classify_category(self, text: str) -> dict:
        """Returns predicted category with confidence scores."""
        text_clean = text.lower().strip()
        probs = self.pipeline.predict_proba([text_clean])[0]
        classes = self.pipeline.classes_
        top_idx = np.argsort(probs)[::-1]
        return {
            "category": classes[top_idx[0]],
            "confidence": round(float(probs[top_idx[0]]), 3),
            "alternatives": [
                {"category": classes[i], "confidence": round(float(probs[i]), 3)}
                for i in top_idx[1:3]
            ],
        }

    def classify_severity(self, text: str) -> str:
        """Rule-based severity detection."""
        text_lower = text.lower()
        for severity in ["Critical", "High", "Medium", "Low"]:
            for kw in SEVERITY_KEYWORDS[severity]:
                if kw in text_lower:
                    return severity
        return "Medium"

    def classify_sentiment(self, text: str) -> str:
        """Rule-based sentiment detection."""
        text_lower = text.lower()
        scores = {}
        for sentiment, keywords in SENTIMENT_KEYWORDS.items():
            scores[sentiment] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Neutral"

    def get_embedding(self, text: str) -> np.ndarray:
        """TF-IDF vector as lightweight embedding for similarity."""
        vec = self.vectorizer.transform([text.lower()])
        return vec.toarray()[0]

    def find_duplicates(self, text: str, complaint_df: pd.DataFrame, threshold=0.45) -> list:
        """Find similar complaints from existing database."""
        if complaint_df.empty:
            return []
        query_vec = self.get_embedding(text)
        corpus_vecs = np.vstack([
            self.get_embedding(t) for t in complaint_df["complaint_text"]
        ])
        sims = cosine_similarity([query_vec], corpus_vecs)[0]
        matches = []
        for idx, sim in enumerate(sims):
            if sim >= threshold:
                row = complaint_df.iloc[idx]
                matches.append({
                    "complaint_id": row["complaint_id"],
                    "similarity": round(float(sim), 3),
                    "status": row["status"],
                    "category": row["category"],
                    "text_snippet": row["complaint_text"][:80] + "...",
                })
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:3]

    def analyze(self, text: str, existing_df: pd.DataFrame = None) -> dict:
        """Full NLP pipeline — category + severity + sentiment + duplicates."""
        category_result = self.classify_category(text)
        severity = self.classify_severity(text)
        sentiment = self.classify_sentiment(text)
        duplicates = []
        if existing_df is not None and not existing_df.empty:
            duplicates = self.find_duplicates(text, existing_df)
        return {
            "category": category_result["category"],
            "confidence": category_result["confidence"],
            "alternatives": category_result["alternatives"],
            "severity": severity,
            "sentiment": sentiment,
            "duplicates": duplicates,
            "is_duplicate": len(duplicates) > 0,
        }


# ---------------------------------------------------------------------------
# Gen-AI response drafting
# ---------------------------------------------------------------------------

RESPONSE_TEMPLATES = {
    "ATM / Debit Card": """Dear {name},

Thank you for reaching out to Union Bank of India regarding your ATM/Debit Card issue (Complaint ID: {cid}).

We sincerely apologise for the inconvenience caused. Your complaint has been registered and assigned to our Cards & Digital Banking team with {severity} priority.

Our team will investigate the issue with your account number {account} and ensure that any erroneous debits are reversed within the stipulated SLA of {sla} hours.

For urgent card blocking, please call 1800-22-2244 (24x7 toll-free).

Warm regards,
Resolve.AI — Union Bank Customer Support""",

    "Net Banking / Mobile App": """Dear {name},

Thank you for contacting Union Bank of India regarding your Digital Banking issue (Complaint ID: {cid}).

We regret the inconvenience you have experienced. Your complaint has been assigned to our Digital Channels Support team with {severity} priority.

Our technical team will investigate and resolve the issue with your account {account} within {sla} hours. If your funds are affected, a reversal will be initiated on priority.

For immediate support, please use our 24x7 helpline: 1800-208-2244.

Warm regards,
Resolve.AI — Union Bank Customer Support""",

    "Fraud / Unauthorized Transaction": """Dear {name},

URGENT — We have received your fraud complaint (Complaint ID: {cid}) and are treating this with the highest priority.

Immediate actions taken:
1. Complaint escalated to our Fraud Investigation Unit
2. Temporary hold placed on suspicious transaction patterns on account {account}
3. Your account is being monitored in real-time

Please do NOT share any OTP, password, or card details with anyone. Our representatives will NEVER ask for these.

If you need immediate assistance, call our Fraud Helpline: 1800-11-0001 (24x7).

We will revert within {sla} hours with investigation findings.

Warm regards,
Resolve.AI — Union Bank Fraud Response Team""",

    "Loan / EMI": """Dear {name},

Thank you for contacting Union Bank of India regarding your Loan/EMI concern (Complaint ID: {cid}).

We apologise for the inconvenience caused. Your complaint regarding account {account} has been forwarded to our Loan Servicing team with {severity} priority.

Our team will review your loan account, verify the concern, and ensure resolution within {sla} hours. Any incorrect deductions will be reversed with applicable interest.

For queries, visit your nearest branch or call 1800-22-2244.

Warm regards,
Resolve.AI — Union Bank Loan Support""",

    "Account / KYC": """Dear {name},

Thank you for writing to Union Bank of India (Complaint ID: {cid}).

Your complaint regarding account {account} has been registered with our Account Services team under {severity} priority. Your KYC documents and account status are being reviewed.

We will resolve this within {sla} hours and keep you informed via WhatsApp/SMS updates.

Warm regards,
Resolve.AI — Union Bank Account Services""",

    "Branch Service": """Dear {name},

Thank you for bringing this to our attention (Complaint ID: {cid}).

We apologise for the unsatisfactory experience at our branch. Your feedback has been forwarded to the Branch Manager and Regional Operations Head under {severity} priority.

We take service quality extremely seriously and will ensure corrective action within {sla} hours. A follow-up call will be made by the branch team.

Warm regards,
Resolve.AI — Union Bank Branch Operations""",
}


def generate_ai_response(
    complaint_text: str,
    category: str,
    severity: str,
    customer_name: str,
    account_number: str,
    complaint_id: str,
    hf_api_key: str = None,
) -> str:
    """
    Generate a response using HuggingFace Inference API if key provided,
    else fall back to a high-quality template-based response.
    """
    sla_map = {"Critical": 4, "High": 24, "Medium": 48, "Low": 72}
    sla = sla_map.get(severity, 48)

    template = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["Account / KYC"])
    template_response = template.format(
        name=customer_name,
        cid=complaint_id,
        account=account_number,
        severity=severity,
        sla=sla,
    )

    if not hf_api_key:
        return template_response

    # Try HuggingFace Inference API with Mistral-7B-Instruct
    try:
        prompt = f"""[INST] You are a professional customer support agent for Union Bank of India. 
Write a concise, empathetic response to this banking complaint in under 150 words.
Be specific about the complaint type: {category}
Severity: {severity}
Customer name: {customer_name}
Complaint: {complaint_text}
Include a complaint reference number: {complaint_id} [/INST]"""

        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.4,
                "return_full_text": False,
            },
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and result:
                ai_text = result[0].get("generated_text", "").strip()
                if len(ai_text) > 50:
                    return ai_text
    except Exception:
        pass

    return template_response

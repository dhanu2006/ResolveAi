"""
generate_data.py
Generates synthetic banking complaint dataset for Resolve.AI POC.
Simulates complaints from WhatsApp, Email, and Web Form channels.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

CHANNELS = ["WhatsApp", "Email", "Web Form"]
COMPLAINT_TEMPLATES = {
    "ATM / Debit Card": [
        "My ATM card was swallowed by the machine and cash was debited but not dispensed.",
        "Unable to withdraw cash from ATM. Transaction failed but amount got deducted.",
        "My debit card is not working at POS terminals since last 3 days.",
        "ATM transaction showed successful but I didn't receive the cash.",
        "My debit card got blocked without any reason. Please unblock it.",
        "Card declined at merchant even though I have sufficient balance.",
    ],
    "Net Banking / Mobile App": [
        "I am unable to login to mobile banking app. Getting invalid credentials error.",
        "Fund transfer failed but amount was deducted from my account.",
        "Net banking session keeps timing out in the middle of transactions.",
        "Unable to register for mobile banking. OTP is not being received.",
        "My transaction history is showing incorrect entries in net banking.",
        "UPI payment failed but money got debited. Please refund immediately.",
    ],
    "Loan / EMI": [
        "My EMI was auto-debited twice this month. Please refund the extra amount.",
        "Loan statement shows incorrect outstanding amount.",
        "I applied for a personal loan 2 weeks ago and have received no response.",
        "Pre-payment charges are being applied even though I opted for a floating rate loan.",
        "My CIBIL score was incorrectly updated by the bank showing loan default.",
        "Home loan sanction letter not received even after 30 days of application.",
    ],
    "Account / KYC": [
        "My account has been frozen without prior notice. I need it urgently.",
        "KYC documents were submitted 3 weeks ago but account is still restricted.",
        "Wrong name spelling on my passbook and debit card.",
        "Unable to update my mobile number linked to bank account.",
        "My account was opened without my consent. Please investigate.",
        "Joint account holder removed without my authorization.",
    ],
    "Fraud / Unauthorized Transaction": [
        "Unauthorized transaction of Rs 45,000 done from my account. I did not do this.",
        "I received a phishing call and my account was debited Rs 12,000 fraudulently.",
        "Multiple small transactions debited from my account which I did not authorize.",
        "My credit card details were stolen and used for online shopping.",
        "OTP was shared unknowingly and Rs 85,000 was transferred out of my account.",
        "Fraudulent FD was opened in my name without my knowledge.",
    ],
    "Branch Service": [
        "Bank staff is extremely rude and unhelpful at the Andheri branch.",
        "Cheque deposited 7 days ago has still not been cleared.",
        "Passbook update machine at branch has been non-functional for 2 weeks.",
        "DD issuance is taking more than 4 days which is unacceptable.",
        "NEFT was submitted at branch but money has not reached beneficiary.",
        "Staff refused to help elderly customer with basic banking needs.",
    ],
}

CATEGORIES = list(COMPLAINT_TEMPLATES.keys())
SEVERITIES = ["Low", "Medium", "High", "Critical"]
SEVERITY_WEIGHTS = [0.25, 0.40, 0.25, 0.10]
SENTIMENTS = ["Angry", "Frustrated", "Neutral", "Concerned"]
SENTIMENT_WEIGHTS = [0.35, 0.35, 0.15, 0.15]
STATUSES = ["Open", "In Progress", "Escalated", "Resolved"]


def generate_complaint_id(index):
    return f"RES-2026-{index + 10001}"


def generate_complaints(n=200):
    complaints = []
    base_date = datetime(2026, 4, 1)

    for i in range(n):
        category = random.choice(CATEGORIES)
        text = random.choice(COMPLAINT_TEMPLATES[category])
        channel = random.choices(
            CHANNELS, weights=[0.55, 0.25, 0.20], k=1
        )[0]
        severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]
        sentiment = random.choices(SENTIMENTS, weights=SENTIMENT_WEIGHTS, k=1)[0]

        # Override severity for fraud complaints
        if category == "Fraud / Unauthorized Transaction":
            severity = random.choices(["High", "Critical"], weights=[0.4, 0.6], k=1)[0]
            sentiment = random.choices(["Angry", "Frustrated"], weights=[0.6, 0.4], k=1)[0]

        created_at = base_date + timedelta(
            days=random.randint(0, 53),
            hours=random.randint(7, 22),
            minutes=random.randint(0, 59),
        )

        sla_hours = {"Low": 72, "Medium": 48, "High": 24, "Critical": 4}[severity]
        status_weights = {
            "Low": [0.20, 0.30, 0.05, 0.45],
            "Medium": [0.25, 0.35, 0.10, 0.30],
            "High": [0.30, 0.30, 0.20, 0.20],
            "Critical": [0.35, 0.25, 0.30, 0.10],
        }
        status = random.choices(STATUSES, weights=status_weights[severity], k=1)[0]

        resolution_time = None
        if status == "Resolved":
            resolution_time = random.randint(1, sla_hours + 24)

        sla_breached = (
            status != "Resolved"
            and (datetime(2026, 5, 24) - created_at).total_seconds() / 3600 > sla_hours
        ) or (
            status == "Resolved"
            and resolution_time is not None
            and resolution_time > sla_hours
        )

        complaints.append(
            {
                "complaint_id": generate_complaint_id(i),
                "customer_name": fake.name(),
                "phone": fake.phone_number(),
                "account_number": f"UBOI{random.randint(100000000, 999999999)}",
                "channel": channel,
                "complaint_text": text,
                "category": category,
                "severity": severity,
                "sentiment": sentiment,
                "status": status,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M"),
                "sla_hours": sla_hours,
                "resolution_time_hrs": resolution_time,
                "sla_breached": sla_breached,
                "agent_assigned": fake.name() if status != "Open" else None,
                "branch": random.choice([
                    "Andheri West", "Pune Camp", "Bandra East", "Thane",
                    "Nagpur Central", "Nashik Road", "Kolhapur", "Aurangabad"
                ]),
            }
        )

    return pd.DataFrame(complaints)


if __name__ == "__main__":
    df = generate_complaints(200)
    df.to_csv("data/complaints.csv", index=False)
    print(f"Generated {len(df)} complaints → data/complaints.csv")
    print(df[["complaint_id", "category", "severity", "channel", "status"]].head(10))

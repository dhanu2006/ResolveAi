"""
app.py — Resolve.AI | Unified Complaint Intelligence Platform
Enterprise prototype — Union Bank of India
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random, os

st.set_page_config(
    page_title="Resolve.AI — Complaint Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from nlp_engine import ResolveAINLPEngine, generate_ai_response
    from generate_data import generate_complaints
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM — clean white enterprise
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f5f6fa;
}
.main { background: #f5f6fa; }
.main .block-container { padding: 0 2rem 2.5rem; max-width: 100%; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 9px 14px;
    border-radius: 7px;
    font-size: 13.5px;
    font-weight: 400;
    color: #4b5563 !important;
    display: block;
    transition: all 0.15s;
    margin-bottom: 1px;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #f3f4f6;
    color: #111827 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    background: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    color: #111827 !important;
    font-size: 12.5px !important;
    border-radius: 7px !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 10px !important;
    color: #9ca3af !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 6px !important;
}

/* ── Top bar ── */
.top-bar {
    background: white;
    border-bottom: 1px solid #e8eaf0;
    padding: 16px 32px;
    margin: 0 -2rem 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.top-bar-brand { display: flex; align-items: center; gap: 12px; }
.brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800; color: white;
}
.brand-name { font-size: 17px; font-weight: 700; color: #111827; letter-spacing: -0.3px; }
.brand-dot  { color: #2563eb; }
.brand-sub  { font-size: 11.5px; color: #6b7280; margin-top: 1px; }
.bank-tag {
    font-size: 11px; font-weight: 600; color: #1d4ed8;
    background: #eff6ff; border: 1px solid #bfdbfe;
    padding: 5px 14px; border-radius: 20px;
}
.page-title { font-size: 15px; font-weight: 600; color: #374151; }
.page-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 24px; }
.kpi {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-blue::after   { background: #2563eb; }
.kpi-red::after    { background: #dc2626; }
.kpi-amber::after  { background: #d97706; }
.kpi-rose::after   { background: #e11d48; }
.kpi-green::after  { background: #16a34a; }
.kpi-purple::after { background: #7c3aed; }
.kpi-n     { font-size: 30px; font-weight: 700; color: #111827; letter-spacing: -1px; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.kpi-l     { font-size: 11.5px; color: #6b7280; margin-top: 5px; font-weight: 500; }
.kpi-d     { font-size: 11px; margin-top: 6px; font-weight: 500; }
.kpi-d-bad  { color: #dc2626; }
.kpi-d-good { color: #16a34a; }

/* ── Section label ── */
.sec {
    font-size: 11.5px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 24px 0 12px;
    display: flex; align-items: center; gap: 10px;
}
.sec::after { content: ''; flex: 1; height: 1px; background: #e5e7eb; }

/* ── Severity / Status pills ── */
.pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.pill-critical { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.pill-high     { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
.pill-medium   { background: #fefce8; color: #854d0e; border: 1px solid #fde68a; }
.pill-low      { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.pill-open       { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.pill-progress   { background: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe; }
.pill-escalated  { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
.pill-resolved   { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

/* ── Complaint row ── */
.crow {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: box-shadow 0.15s, border-color 0.15s;
}
.crow:hover { border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.08); }
.crow-breach { border-left: 3px solid #dc2626; }
.c-id   { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #2563eb; font-weight: 500; }
.c-text { font-size: 13.5px; color: #374151; margin: 5px 0 4px; line-height: 1.5; }
.c-meta { font-size: 11.5px; color: #9ca3af; }
.c-sla  { font-size: 11px; color: #dc2626; font-weight: 600; }

/* ── Chart wrapper ── */
.chart-card {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.chart-label { font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px; }

/* ── AI box ── */
.ai-box {
    background: #f8faff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 16px 18px;
    font-size: 13.5px;
    color: #374151;
    line-height: 1.75;
    white-space: pre-wrap;
    margin-top: 12px;
}

/* ── NLP chips ── */
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 6px; }
.chip {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px 14px;
    min-width: 90px;
}
.chip-l { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.chip-v { font-size: 15px; font-weight: 700; color: #111827; margin-top: 2px; }

/* ── Streamlit element overrides ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 7px 18px !important;
    border: 1px solid #d1d5db !important;
    background: white !important;
    color: #374151 !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover { border-color: #2563eb !important; color: #2563eb !important; background: #eff6ff !important; }
.stButton > button[kind="primary"] {
    background: #2563eb !important; border-color: #2563eb !important;
    color: white !important; box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="primary"]:hover { background: #1d4ed8 !important; border-color: #1d4ed8 !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 13.5px !important;
}
label, .stMarkdown p { color: #374151 !important; font-size: 13px !important; }
.stProgress > div > div { background: #2563eb !important; }
.stProgress > div { background: #e5e7eb !important; border-radius: 99px !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 7px 18px;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #111827 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.stExpander {
    background: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}
.stExpander summary { font-size: 13.5px !important; color: #374151 !important; }
.stSuccess, .stInfo, .stWarning, .stError { border-radius: 8px !important; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #111827 !important; }
div[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
h1,h2,h3 { color: #111827 !important; font-family: 'Inter', sans-serif !important; }

/* ── NFC card ── */
@keyframes card-float {
    0%,100% { transform: translateY(0) rotateX(0deg); }
    50%      { transform: translateY(-6px) rotateX(2deg); }
}
@keyframes nfc-ripple {
    0%   { transform: scale(1); opacity: 0.6; }
    100% { transform: scale(2.5); opacity: 0; }
}
@keyframes step-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(37,99,235,0.2); }
    50%      { box-shadow: 0 0 0 6px rgba(37,99,235,0); }
}
.nfc-card {
    width: 360px; height: 224px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1a1208 0%, #2d2010 25%, #4a3820 50%, #2d2010 75%, #1a1208 100%);
    position: relative; overflow: hidden;
    box-shadow:
        0 24px 64px rgba(0,0,0,0.35),
        0 0 0 1px rgba(212,175,55,0.25),
        inset 0 1px 0 rgba(212,175,55,0.4),
        inset 0 -1px 0 rgba(0,0,0,0.3);
    animation: card-float 4s ease-in-out infinite;
    cursor: pointer;
    transition: transform 0.3s;
}
.nfc-card:hover { animation-play-state: paused; transform: scale(1.02); }
.card-shine {
    position: absolute; inset: 0;
    background: linear-gradient(125deg, rgba(255,255,255,0.10) 0%, transparent 40%, rgba(255,255,255,0.04) 100%);
    pointer-events: none;
}
.card-pattern {
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        -45deg,
        transparent, transparent 8px,
        rgba(212,175,55,0.025) 8px, rgba(212,175,55,0.025) 9px
    );
    pointer-events: none;
}
.card-chip {
    position: absolute; top: 64px; left: 26px;
    width: 42px; height: 32px;
    background: linear-gradient(135deg, #c8a84b 0%, #f0d878 40%, #c8a84b 70%, #e8c860 100%);
    border-radius: 5px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.4), 0 2px 4px rgba(0,0,0,0.5);
}
.card-chip::before {
    content: '';
    position: absolute;
    inset: 5px;
    border: 1px solid rgba(120,80,0,0.4);
    border-radius: 2px;
}
.card-chip::after {
    content: '';
    position: absolute;
    left: 50%; top: 5px; bottom: 5px;
    width: 1px; margin-left: -0.5px;
    background: rgba(120,80,0,0.25);
}
.card-nfc {
    position: absolute; top: 58px; right: 26px;
    display: flex; flex-direction: column; align-items: center; gap: 2px;
}
.nfc-arc {
    border: 2px solid transparent;
    border-left-color: rgba(212,175,55,0.65);
    border-radius: 50%;
    transform: rotate(-45deg);
}
.card-priority {
    position: absolute; top: 18px; left: 26px;
    font-size: 9px; font-weight: 700; color: rgba(212,175,55,0.75);
    text-transform: uppercase; letter-spacing: 0.16em;
    border: 1px solid rgba(212,175,55,0.3);
    padding: 2px 9px; border-radius: 10px;
}
.card-brand {
    position: absolute; top: 16px; right: 26px;
    font-size: 13px; font-weight: 800; color: #d4af37;
    letter-spacing: -0.3px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
.card-brand span { color: rgba(212,175,55,0.45); }
.card-name {
    position: absolute; bottom: 40px; left: 26px;
    font-size: 14px; font-weight: 500; color: rgba(255,255,255,0.88);
    letter-spacing: 0.06em; text-transform: uppercase;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
}
.card-bank {
    position: absolute; bottom: 20px; left: 26px;
    font-size: 9.5px; font-weight: 600; color: rgba(212,175,55,0.7);
    letter-spacing: 0.14em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.card-num {
    position: absolute; bottom: 20px; right: 26px;
    font-size: 10px; color: rgba(212,175,55,0.55);
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.1em;
}

/* ── NFC steps ── */
.step-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; border-radius: 10px;
    border: 1px solid #e5e7eb;
    background: white;
    margin-bottom: 8px;
    transition: all 0.3s;
}
.step-row.active {
    border-color: #2563eb;
    background: #eff6ff;
    box-shadow: 0 2px 8px rgba(37,99,235,0.12);
    animation: step-pulse 1.5s ease-in-out infinite;
}
.step-row.done { border-color: #bbf7d0; background: #f0fdf4; }
.step-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: #f3f4f6; display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #9ca3af; flex-shrink: 0;
}
.step-num.active { background: #2563eb; color: white; }
.step-num.done   { background: #16a34a; color: white; }
.step-t { font-size: 13px; font-weight: 500; color: #111827; }
.step-s { font-size: 11.5px; color: #6b7280; margin-top: 2px; }

/* ── WhatsApp phone ── */
.phone-wrap { display: flex; justify-content: center; }
.phone-outer {
    width: 310px;
    background: #1a1a2a;
    border-radius: 38px;
    padding: 10px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.2), 0 0 0 1px #2a2a3e, inset 0 0 0 2px #0e0e1a;
}
.phone-screen {
    background: #f0ebe3;
    border-radius: 30px;
    overflow: hidden;
    min-height: 520px;
    display: flex; flex-direction: column;
}
.wa-bar {
    background: #075e54;
    padding: 10px 14px 8px;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
}
.wa-av {
    width: 36px; height: 36px; border-radius: 50%;
    background: #128c7e;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: white; flex-shrink: 0;
}
.wa-n { font-size: 14px; font-weight: 600; color: white; line-height: 1.2; }
.wa-s { font-size: 10.5px; color: rgba(255,255,255,0.7); }
.wa-chat { flex: 1; padding: 10px 8px; overflow-y: auto; }
.wa-in {
    background: white;
    border-radius: 2px 12px 12px 12px;
    padding: 7px 11px 5px;
    margin: 5px 36px 5px 4px;
    font-size: 12px; color: #111; line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    max-width: 85%;
}
.wa-out {
    background: #dcf8c6;
    border-radius: 12px 2px 12px 12px;
    padding: 7px 11px 5px;
    margin: 5px 4px 5px auto;
    font-size: 12px; color: #111; line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    max-width: 85%;
}
.wa-t { font-size: 10px; color: #8a9a85; text-align: right; margin-top: 2px; }
.wa-ticket {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 5px 4px 5px 4px;
    font-size: 11px; color: #1e40af;
}
.wa-ticket-header { font-weight: 700; color: #1d4ed8; margin-bottom: 6px; font-size: 11.5px; }
.wa-ticket-row { color: #374151; margin-bottom: 3px; }
.wa-ticket-sev { font-weight: 700; color: #dc2626; }
.wa-status-pill {
    display: inline-block; background: #dbeafe; color: #1d4ed8;
    font-size: 10px; font-weight: 700; padding: 2px 8px;
    border-radius: 10px; margin-top: 6px;
}

/* ── Info card ── */
.info-card {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 12px;
}
.info-card-title { font-size: 13.5px; font-weight: 600; color: #111827; margin-bottom: 4px; }
.info-card-body  { font-size: 13px; color: #6b7280; line-height: 1.6; }
.metric-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #f3f4f6; }
.metric-row:last-child { border: none; }
.metric-label { font-size: 12.5px; color: #6b7280; }
.metric-val   { font-size: 13px; font-weight: 600; color: #2563eb; }

/* ── Insight box ── */
.insight {
    background: #f8faff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 13.5px;
    color: #374151;
    line-height: 1.75;
    margin-top: 20px;
}
.insight-title { font-size: 12px; font-weight: 600; color: #2563eb; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP engine…")
def load_engine():
    return ResolveAINLPEngine()

@st.cache_data(show_spinner="Loading complaint database…")
def load_data():
    os.makedirs("data", exist_ok=True)
    df = generate_complaints(200)
    df.to_csv("data/complaints.csv", index=False)
    return df

engine = load_engine()
df     = load_data()

SLA_MAP    = {"Critical": 4, "High": 24, "Medium": 48, "Low": 72}
SEV_COLOR  = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#16a34a"}
SENT_COLOR = {"Angry": "#dc2626", "Frustrated": "#ea580c", "Concerned": "#ca8a04", "Neutral": "#6b7280"}

def sev_pill(s):
    return f"<span class='pill pill-{s.lower()}'>{s}</span>"
def sta_pill(s):
    cls = s.lower().replace(" ", "-").replace("in-progress","progress")
    return f"<span class='pill pill-{cls}'>{s}</span>"

def light_layout(fig, height=290):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#6b7280", size=12),
        margin=dict(l=0, r=0, t=8, b=8),
        legend=dict(bgcolor="white", font_color="#6b7280"),
        xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
        yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:22px 12px 18px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px;'>
        <div style='font-size:21px; font-weight:800; color:#111827; letter-spacing:-0.5px;'>Resolve<span style='color:#2563eb;'>.</span>AI</div>
        <div style='font-size:11px; color:#6b7280; margin-top:3px;'>Complaint Intelligence Platform</div>
        <div style='margin-top:10px; display:inline-block; background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; font-size:10px; font-weight:600; padding:3px 10px; border-radius:10px; letter-spacing:0.04em;'>Union Bank of India</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("NAVIGATION")
    page = st.radio("nav", [
        "Command Centre",
        "Live Complaint Intake",
        "Complaint Explorer",
        "Root Cause Intelligence",
        "NFC Premium Channel",
        "WhatsApp Journey Demo",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("CONFIGURATION")
    hf_key = st.text_input("HuggingFace API Token", type="password",
                            placeholder="hf_…", label_visibility="visible")
    if hf_key:
        st.markdown("<div style='font-size:11px; color:#16a34a; margin-top:4px;'>Enhanced AI responses active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:11px; color:#6b7280; margin-top:4px;'>Template responses active</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("SYSTEM STATUS")
    st.markdown("""
    <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px;'>
        <div style='font-size:11.5px; color:#166534; display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
            <span style='width:7px; height:7px; background:#16a34a; border-radius:50%; flex-shrink:0;'></span>NLP Engine Online
        </div>
        <div style='font-size:11.5px; color:#166534; display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
            <span style='width:7px; height:7px; background:#16a34a; border-radius:50%; flex-shrink:0;'></span>Database Connected
        </div>
        <div style='font-size:11.5px; color:#92400e; display:flex; align-items:center; gap:8px;'>
            <span style='width:7px; height:7px; background:#d97706; border-radius:50%; flex-shrink:0;'></span>WhatsApp API — Simulated
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────
PAGE_META = {
    "Command Centre":          ("Command Centre",          "Real-time operations overview — complaint volume, SLA status and team workload"),
    "Live Complaint Intake":   ("Live Complaint Intake",   "Ingest complaints from any channel with AI-powered classification and response drafting"),
    "Complaint Explorer":      ("Complaint Explorer",      "360-degree view of every complaint — filter, investigate and action from one place"),
    "Root Cause Intelligence": ("Root Cause Intelligence", "Detect systemic patterns, predict complaint spikes and identify operational failures"),
    "NFC Premium Channel":     ("NFC Premium Channel",     "One-tap complaint registration for priority customers via NFC SmartCard"),
    "WhatsApp Journey Demo":   ("WhatsApp Journey Demo",   "Live simulation of the complete customer complaint journey via WhatsApp"),
}
title, subtitle = PAGE_META[page]
st.markdown(f"""
<div class='top-bar'>
    <div class='top-bar-brand'>
        <div class='brand-icon'>R</div>
        <div>
            <div class='brand-name'>Resolve<span class='brand-dot'>.</span>AI</div>
            <div class='brand-sub'>Unified Complaint Intelligence</div>
        </div>
    </div>
    <div style='text-align:center;'>
        <div class='page-title'>{title}</div>
        <div class='page-sub'>{subtitle}</div>
    </div>
    <div class='bank-tag'>Union Bank of India — Production Prototype</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTRE
# ═══════════════════════════════════════════════════════════════
if page == "Command Centre":
    total     = len(df)
    open_c    = len(df[df["status"] == "Open"])
    escalated = len(df[df["status"] == "Escalated"])
    breached  = int(df["sla_breached"].sum())
    resolved  = len(df[df["status"] == "Resolved"])
    fraud_c   = len(df[df["category"] == "Fraud / Unauthorized Transaction"])

    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi kpi-blue'>
            <div class='kpi-n'>{total}</div>
            <div class='kpi-l'>Total Complaints</div>
            <div class='kpi-d kpi-d-bad'>+12 this week</div>
        </div>
        <div class='kpi kpi-red'>
            <div class='kpi-n'>{open_c}</div>
            <div class='kpi-l'>Open</div>
            <div class='kpi-d kpi-d-bad'>Requires action</div>
        </div>
        <div class='kpi kpi-amber'>
            <div class='kpi-n'>{escalated}</div>
            <div class='kpi-l'>Escalated</div>
            <div class='kpi-d kpi-d-bad'>{int(escalated/max(total,1)*100)}% of total</div>
        </div>
        <div class='kpi kpi-rose'>
            <div class='kpi-n'>{breached}</div>
            <div class='kpi-l'>SLA Breached</div>
            <div class='kpi-d kpi-d-bad'>Immediate review</div>
        </div>
        <div class='kpi kpi-green'>
            <div class='kpi-n'>{resolved}</div>
            <div class='kpi-l'>Resolved</div>
            <div class='kpi-d kpi-d-good'>{int(resolved/max(total,1)*100)}% rate</div>
        </div>
        <div class='kpi kpi-purple'>
            <div class='kpi-n'>{fraud_c}</div>
            <div class='kpi-l'>Fraud Alerts</div>
            <div class='kpi-d kpi-d-bad'>Critical priority</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        cat_c = df["category"].value_counts().reset_index()
        cat_c.columns = ["Cat","Count"]
        cat_c["Short"] = cat_c["Cat"].str.split("/").str[0].str.strip()
        fig = px.bar(cat_c, x="Count", y="Short", orientation="h",
                     color="Count", color_continuous_scale=[[0,"#bfdbfe"],[1,"#1d4ed8"]],
                     text="Count")
        fig.update_traces(textposition="outside", textfont_color="#6b7280",
                          textfont_size=11, marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig = light_layout(fig, 280)
        fig.update_xaxes(title="", showgrid=False)
        fig.update_yaxes(title="")
        st.markdown("<div class='chart-card'><div class='chart-label'>Complaints by Category</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        sev = df["severity"].value_counts()
        fig2 = px.pie(values=sev.values, names=sev.index,
                      color=sev.index, color_discrete_map=SEV_COLOR, hole=0.52)
        fig2.update_traces(textposition="outside", textinfo="percent+label",
                           marker_line_color="white", marker_line_width=2)
        fig2 = light_layout(fig2, 280)
        fig2.update_layout(showlegend=False)
        st.markdown("<div class='chart-card'><div class='chart-label'>Severity Split</div>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        ch = df.groupby("channel")["complaint_id"].count().reset_index()
        ch.columns = ["Channel","Count"]
        ch_col = {"WhatsApp":"#25d366","Email":"#2563eb","Web Form":"#ea580c"}
        fig3 = px.bar(ch, x="Channel", y="Count", color="Channel",
                      color_discrete_map=ch_col, text="Count")
        fig3.update_traces(textposition="outside", textfont_color="#6b7280",
                           textfont_size=11, marker_line_width=0)
        fig3 = light_layout(fig3, 280)
        fig3.update_layout(showlegend=False)
        fig3.update_xaxes(title="")
        fig3.update_yaxes(title="", showgrid=False)
        st.markdown("<div class='chart-card'><div class='chart-label'>Intake Channel</div>", unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c4, c5 = st.columns([1.6, 1])
    with c4:
        df_t = df.copy()
        df_t["date"] = pd.to_datetime(df_t["created_at"]).dt.date
        td = df_t.groupby("date")["complaint_id"].count().reset_index()
        td.columns = ["Date","Count"]
        fig4 = px.area(td, x="Date", y="Count", color_discrete_sequence=["#2563eb"])
        fig4.update_traces(fillcolor="rgba(37,99,235,0.08)", line_color="#2563eb", line_width=2)
        fig4 = light_layout(fig4, 230)
        fig4.update_xaxes(title="")
        fig4.update_yaxes(title="")
        st.markdown("<div class='chart-card'><div class='chart-label'>Daily Complaint Volume Trend</div>", unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        sent = df["sentiment"].value_counts().reset_index()
        sent.columns = ["Sentiment","Count"]
        fig5 = px.bar(sent, x="Sentiment", y="Count", color="Sentiment",
                      color_discrete_map=SENT_COLOR, text="Count")
        fig5.update_traces(textposition="outside", textfont_color="#6b7280",
                           textfont_size=11, marker_line_width=0)
        fig5 = light_layout(fig5, 230)
        fig5.update_layout(showlegend=False)
        fig5.update_xaxes(title="")
        fig5.update_yaxes(title="", showgrid=False)
        st.markdown("<div class='chart-card'><div class='chart-label'>Sentiment Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Priority Queue — Requires Immediate Attention</div>", unsafe_allow_html=True)
    sev_rank = {"Critical":0,"High":1,"Medium":2,"Low":3}
    urgent = df[df["status"].isin(["Open","Escalated"])].copy()
    urgent["r"] = urgent["severity"].map(sev_rank)
    for _, row in urgent.sort_values(["r","created_at"]).head(8).iterrows():
        bc = "crow crow-breach" if row["sla_breached"] else "crow"
        slw = "<span class='c-sla'>SLA BREACH</span>" if row["sla_breached"] else ""
        st.markdown(f"""
        <div class='{bc}'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div style='flex:1;'>
                    <span class='c-id'>{row['complaint_id']}</span>
                    <span class='c-meta' style='margin-left:10px;'>{row['channel']}  ·  {str(row['created_at'])[:16]}</span>
                    <div class='c-text'>{row['complaint_text'][:110]}…</div>
                    <div class='c-meta'>{row['customer_name']}  ·  {row['branch']}  ·  SLA: {row['sla_hours']}h  {slw}</div>
                </div>
                <div style='text-align:right; flex-shrink:0; margin-left:16px;'>
                    {sev_pill(row['severity'])}
                    <div style='margin-top:6px;'>{sta_pill(row['status'])}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — LIVE COMPLAINT INTAKE
# ═══════════════════════════════════════════════════════════════
elif page == "Live Complaint Intake":
    SAMPLES = [
        ("ATM — cash not dispensed",
         "My ATM card was swallowed by the machine at Bandra branch and Rs 10,000 was debited from my account but cash was not dispensed. Transaction reference UBI2026049821."),
        ("Fraud — unauthorised transfer",
         "An unauthorised transfer of Rs 45,000 was made from my savings account at 2:47am. I did not authorise this transaction and received no OTP. Please freeze my account immediately."),
        ("Net banking — login failure",
         "I have been unable to login to the mobile banking app for the past 4 days. The app shows a session timeout error immediately after OTP verification."),
        ("Loan — EMI debited twice",
         "My home loan EMI of Rs 28,500 was debited twice on the same day this month. This has caused my account balance to go negative and I have been charged overdraft fees."),
        ("Branch — poor service",
         "The staff at Nagpur Central branch refused to assist my elderly mother with a demand draft request and told her to come back the next day without any explanation."),
        ("KYC — account frozen",
         "My account has been frozen without any prior notice or communication. I submitted updated KYC documents at the branch three weeks ago and urgently need access to my funds."),
    ]

    lc, rc = st.columns([1, 1.15])
    with lc:
        st.markdown("<div class='sec'>Complaint Details</div>", unsafe_allow_html=True)
        channel  = st.selectbox("Intake Channel", ["WhatsApp","Email","Web Form","NFC SmartCard","Branch"])
        cust     = st.text_input("Customer Name", value="Rahul Sharma")
        acct     = st.text_input("Account Number", value="UBOI123456789")
        ctext    = st.text_area("Complaint Description", height=130,
                                placeholder="Paste or type the customer complaint here…",
                                value=st.session_state.get("_fill",""))

        st.markdown("<div class='sec'>Sample Complaints — Click to Load</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (lbl, smp) in enumerate(SAMPLES):
            if cols[i % 2].button(lbl, key=f"smp{i}", use_container_width=True):
                st.session_state["_fill"] = smp
                st.rerun()
        go = st.button("Analyse and Register Complaint", type="primary", use_container_width=True)

    with rc:
        st.markdown("<div class='sec'>AI Analysis Output</div>", unsafe_allow_html=True)
        if go and ctext.strip():
            with st.spinner("Running NLP pipeline…"):
                res    = engine.analyze(ctext, df)
                new_id = f"RES-2026-{random.randint(11000,12999)}"
                sla    = SLA_MAP[res["severity"]]
                resp   = generate_ai_response(
                    ctext, res["category"], res["severity"],
                    cust, acct, new_id, hf_key or None)
            st.success(f"Complaint registered — {new_id}")

            conf = int(res["confidence"] * 100)
            sev_c = SEV_COLOR[res["severity"]]
            st.markdown(f"""
            <div class='chip-row'>
                <div class='chip'><div class='chip-l'>Category</div><div class='chip-v' style='font-size:13px;'>{res['category'].split('/')[0].strip()}</div></div>
                <div class='chip'><div class='chip-l'>Severity</div><div class='chip-v' style='color:{sev_c};'>{res['severity']}</div></div>
                <div class='chip'><div class='chip-l'>Sentiment</div><div class='chip-v' style='font-size:13px;'>{res['sentiment']}</div></div>
                <div class='chip'><div class='chip-l'>Confidence</div><div class='chip-v'>{conf}%</div></div>
                <div class='chip'><div class='chip-l'>SLA Target</div><div class='chip-v'>{sla}h</div></div>
            </div>""", unsafe_allow_html=True)
            st.progress(res["confidence"])

            if res["alternatives"]:
                alts = "  ·  ".join(f"{a['category'].split('/')[0].strip()} ({int(a['confidence']*100)}%)" for a in res["alternatives"])
                st.caption(f"Alternative categories: {alts}")

            if res["is_duplicate"]:
                st.warning(f"{len(res['duplicates'])} similar complaint(s) detected — possible systemic issue")
                for d in res["duplicates"]:
                    st.markdown(f"- `{d['complaint_id']}` ({d['status']}) · {int(d['similarity']*100)}% match · {d['text_snippet']}")
            else:
                st.markdown("<div style='font-size:12px; color:#16a34a; margin:6px 0 2px;'>No duplicate complaints found</div>", unsafe_allow_html=True)

            st.markdown("<div class='sec'>AI-Drafted Customer Response</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

            a1,a2,a3 = st.columns(3)
            a1.button("Send via WhatsApp",  use_container_width=True)
            a2.button("Assign to Agent",    use_container_width=True)
            a3.button("Escalate",           use_container_width=True)

        elif go:
            st.warning("Please enter or select a complaint description.")
        else:
            st.markdown("""
            <div class='chart-card' style='text-align:center; padding:52px 32px;'>
                <div style='font-size:32px; color:#d1d5db;'>&#8635;</div>
                <div style='font-size:14px; color:#9ca3af; margin-top:10px; font-weight:500;'>Fill in complaint details and click Analyse</div>
                <div style='font-size:13px; color:#d1d5db; margin-top:6px; line-height:1.7;'>
                    The NLP pipeline will classify the complaint, score severity<br>
                    and sentiment, check for duplicates, and draft a response.
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — COMPLAINT EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "Complaint Explorer":
    f1,f2,f3,f4 = st.columns(4)
    sch  = f1.multiselect("Channel",  df["channel"].unique(), default=list(df["channel"].unique()))
    ssev = f2.multiselect("Severity", ["Critical","High","Medium","Low"], default=["Critical","High","Medium","Low"])
    ssta = f3.multiselect("Status",   df["status"].unique(), default=list(df["status"].unique()))
    scat = f4.multiselect("Category", df["category"].unique(), default=list(df["category"].unique()))

    filt = df[df["channel"].isin(sch) & df["severity"].isin(ssev) &
              df["status"].isin(ssta) & df["category"].isin(scat)]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Filtered Complaints", len(filt))
    m2.metric("SLA Breaches", int(filt["sla_breached"].sum()))
    avg = filt["resolution_time_hrs"].dropna().mean()
    m3.metric("Avg Resolution", f"{avg:.0f}h" if not np.isnan(avg) else "—")
    m4.metric("Critical Open", len(filt[(filt["severity"]=="Critical") & filt["status"].isin(["Open","Escalated"])]))

    st.markdown(f"<div class='sec'>Showing {len(filt)} complaints</div>", unsafe_allow_html=True)
    for _, row in filt.sort_values("created_at", ascending=False).head(30).iterrows():
        with st.expander(f"{row['complaint_id']}  —  {row['category']}  —  {row['customer_name']}  —  {row['severity']}"):
            d1,d2 = st.columns([1.6,1])
            with d1:
                st.markdown(f"**Complaint**")
                st.markdown(f"<div class='c-text' style='font-size:14px;'>{row['complaint_text']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Customer:** {row['customer_name']}  ·  **Account:** `{row['account_number']}`  ·  **Branch:** {row['branch']}")
            with d2:
                st.markdown(f"""
                <div class='info-card'>
                    <div style='display:flex; gap:8px; margin-bottom:10px;'>{sev_pill(row['severity'])} {sta_pill(row['status'])}</div>
                    <div class='c-meta' style='line-height:2;'>
                        Channel: {row['channel']}<br>Created: {str(row['created_at'])[:16]}<br>
                        Sentiment: {row['sentiment']}<br>
                        SLA: {row['sla_hours']}h — {'<span class="c-sla">BREACHED</span>' if row['sla_breached'] else '<span style="color:#16a34a;font-weight:500;">On track</span>'}<br>
                        Agent: {row['agent_assigned'] or 'Unassigned'}
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button("Generate AI Response", key=f"gen_{row['complaint_id']}", use_container_width=True):
                    with st.spinner("Drafting…"):
                        r = generate_ai_response(
                            row["complaint_text"], row["category"], row["severity"],
                            row["customer_name"], row["account_number"],
                            row["complaint_id"], hf_key or None)
                    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ROOT CAUSE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif page == "Root Cause Intelligence":
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Recurring Patterns", "SLA Heatmap"])

    with tab1:
        df_t = df.copy()
        df_t["date"] = pd.to_datetime(df_t["created_at"]).dt.date
        td = df_t.groupby("date")["complaint_id"].count().reset_index()
        td.columns = ["Date","Count"]
        fig = px.area(td, x="Date", y="Count", color_discrete_sequence=["#2563eb"])
        fig.update_traces(fillcolor="rgba(37,99,235,0.07)", line_color="#2563eb", line_width=2)
        fig = light_layout(fig, 320)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Complaints per day")
        st.markdown("<div class='chart-card'><div class='chart-label'>Daily Complaint Volume</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        daily = df_t.groupby("date")["complaint_id"].count()
        mu, sigma = daily.mean(), daily.std()
        spikes = daily[daily > mu + 1.5*sigma]
        if not spikes.empty:
            st.markdown(f"""
            <div style='background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626; border-radius:10px; padding:14px 20px; margin-top:12px;'>
                <div style='font-size:13px; font-weight:600; color:#991b1b;'>Complaint spike detected on {len(spikes)} day(s)</div>
                <div style='font-size:12.5px; color:#6b7280; margin-top:4px;'>Dates: {", ".join(str(d) for d in spikes.index)}  ·  Threshold: {mu+1.5*sigma:.1f} complaints/day</div>
                <div style='font-size:12.5px; color:#6b7280; margin-top:4px;'>Recommended action: activate surge-response protocol and cross-reference with known service incidents</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.success("No significant complaint spikes detected in the current period.")

    with tab2:
        rec = df.groupby(["category","severity"])["complaint_id"].count().reset_index()
        rec.columns = ["Category","Severity","Count"]
        rec["Short"] = rec["Category"].str.split("/").str[0].str.strip()
        fig2 = px.bar(rec, x="Short", y="Count", color="Severity",
                      color_discrete_map=SEV_COLOR, barmode="stack", text_auto=True)
        fig2.update_traces(textfont_color="rgba(0,0,0,0.5)", textfont_size=10, marker_line_width=0)
        fig2 = light_layout(fig2, 320)
        fig2.update_xaxes(title="", tickangle=-15)
        fig2.update_yaxes(title="")
        st.markdown("<div class='chart-card'><div class='chart-label'>Complaint Volume by Category and Severity</div>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec'>Systemic Issues — High and Critical Volume</div>", unsafe_allow_html=True)
        top = df[df["severity"].isin(["Critical","High"])].groupby("category")["complaint_id"].count().sort_values(ascending=False)
        for cat, cnt in top.items():
            pct = int(cnt/max(len(df),1)*100)
            short = cat.split("/")[0].strip()
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:10px;'>
                <div style='font-size:13px; color:#374151; width:190px; font-weight:500;'>{short}</div>
                <div style='flex:1; background:#f3f4f6; border-radius:99px; height:7px;'>
                    <div style='background:#2563eb; width:{pct}%; height:7px; border-radius:99px;'></div>
                </div>
                <div style='font-size:12px; color:#6b7280; width:90px; text-align:right;'>{cnt} complaints</div>
            </div>""", unsafe_allow_html=True)

    with tab3:
        heat = df.groupby(["category","severity"])["sla_breached"].mean().reset_index()
        heat["sla_breached"] = (heat["sla_breached"]*100).round(1)
        hp = heat.pivot(index="category", columns="severity", values="sla_breached").fillna(0)
        for c in ["Critical","High","Medium","Low"]:
            if c not in hp.columns:
                hp[c] = 0
        hp = hp[["Critical","High","Medium","Low"]]
        hp.index = [i.split("/")[0].strip() for i in hp.index]
        fig3 = px.imshow(hp, color_continuous_scale=[[0,"#eff6ff"],[0.5,"#bfdbfe"],[1,"#dc2626"]],
                         aspect="auto", text_auto=".1f")
        fig3.update_traces(textfont_color="#374151", textfont_size=12)
        fig3 = light_layout(fig3, 320)
        st.markdown("<div class='chart-card'><div class='chart-label'>SLA Breach Rate (%) by Category and Severity</div>", unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Darker red = higher percentage of complaints in that segment that breached their SLA deadline")

    breach_pct = len(df[df["sla_breached"]==True])/max(len(df),1)*100
    fraud_pct  = len(df[df["category"]=="Fraud / Unauthorized Transaction"])/max(len(df),1)*100
    st.markdown(f"""
    <div class='insight'>
        <div class='insight-title'>AI Insight Summary</div>
        Overall SLA breach rate is <strong>{breach_pct:.1f}%</strong> — correlates with staffing gaps during 9–11am and 6–8pm peaks.
        Fraud complaints represent <strong>{fraud_pct:.1f}%</strong> of volume but carry the highest severity and regulatory exposure.
        Recommended action: implement auto-escalation for any Critical complaint unresolved after 2 hours, and deploy dedicated NLP triage for ATM and Net Banking complaints which account for the highest recurring volume.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — NFC PREMIUM CHANNEL
# ═══════════════════════════════════════════════════════════════
elif page == "NFC Premium Channel":

    # ── Session state for step-by-step advance ──
    if "nfc_step" not in st.session_state:
        st.session_state["nfc_step"] = 0

    STEPS = [
        ("Card Detected",       "NFC reader identifies NTAG215 chip — customer account retrieved instantly"),
        ("Identity Verified",   "Account number and registered mobile matched — OTP pre-authorised"),
        ("WhatsApp Opens",      "Deep-link triggers WhatsApp with customer context and account details pre-filled"),
        ("Complaint Submitted", "Customer describes issue — message routed to Resolve.AI processing engine"),
        ("AI Classification",   "NLP pipeline: category identified, severity Critical, sentiment Angry"),
        ("Agent Assigned",      "Complaint routed to Fraud Investigation Unit — SLA clock started at 4 hours"),
        ("Customer Notified",   "Acknowledgement and ticket ID RES-2026-11482 sent back via WhatsApp"),
    ]

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("""
        <div style='margin-bottom:20px;'>
            <div style='font-size:20px; font-weight:700; color:#92400e; letter-spacing:-0.3px;'>Priority SmartCard</div>
            <div style='font-size:13.5px; color:#6b7280; margin-top:5px; line-height:1.7;'>
                One tap. Zero friction. Complaint registered in under 5 seconds.<br>
                Issued exclusively to premium and high-value customers.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Golden NFC card — pure HTML/CSS, no JS interaction needed
        st.markdown("""
        <div style='display:flex; justify-content:center; margin:8px 0 28px;'>
            <div class='nfc-card'>
                <div class='card-shine'></div>
                <div class='card-pattern'></div>
                <div class='card-priority'>Priority Customer</div>
                <div class='card-brand'>Resolve<span>.</span>AI</div>
                <div class='card-chip'></div>
                <div class='card-nfc'>
                    <div class='nfc-arc' style='width:30px;height:30px;'></div>
                    <div class='nfc-arc' style='width:22px;height:22px;margin-top:-22px;'></div>
                    <div class='nfc-arc' style='width:14px;height:14px;margin-top:-22px;'></div>
                </div>
                <div class='card-name'>Rahul Sharma</div>
                <div class='card-bank'>Union Bank of India</div>
                <div class='card-num'>**** **** **** 4821</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        specs = [
            ("NFC Chip",        "NTAG213/215 — compatible with all NFC-enabled smartphones"),
            ("Registration",    "Complaint logged in under 5 seconds"),
            ("No app required", "Opens WhatsApp directly — zero login, zero download"),
            ("Priority routing","NFC complaints auto-queued at Critical priority"),
            ("Issued to",       "Priority, Premium, and High-Value Account holders"),
        ]
        for label, desc in specs:
            st.markdown(f"""
            <div style='display:flex; gap:14px; padding:10px 0; border-bottom:1px solid #f3f4f6;'>
                <div style='font-size:12px; font-weight:600; color:#92400e; min-width:130px;'>{label}</div>
                <div style='font-size:12.5px; color:#6b7280;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Advance buttons — one step at a time, reliable Streamlit state
        b1, b2 = st.columns(2)
        tap_btn   = b1.button("Simulate Card Tap", type="primary", use_container_width=True,
                               disabled=(st.session_state["nfc_step"] > 0 and st.session_state["nfc_step"] <= len(STEPS)))
        next_btn  = b2.button("Next Step",         use_container_width=True,
                               disabled=(st.session_state["nfc_step"] == 0 or st.session_state["nfc_step"] > len(STEPS)))
        reset_btn = b1.button("Reset Demo",        use_container_width=True,
                               disabled=(st.session_state["nfc_step"] == 0))

        if tap_btn:
            st.session_state["nfc_step"] = 1
            st.rerun()
        if next_btn and 0 < st.session_state["nfc_step"] <= len(STEPS):
            st.session_state["nfc_step"] += 1
            st.rerun()
        if reset_btn:
            st.session_state["nfc_step"] = 0
            st.rerun()

    with right:
        st.markdown("<div class='sec'>Journey — Step by Step</div>", unsafe_allow_html=True)

        step = st.session_state["nfc_step"]
        for i, (title, sub) in enumerate(STEPS, 1):
            if step == 0:
                cls, ncls = "step-row", ""
                icon = str(i)
            elif i < step:
                cls, ncls = "step-row done", "done"
                icon = "&#10003;"
            elif i == step:
                cls, ncls = "step-row active", "active"
                icon = str(i)
            else:
                cls, ncls = "step-row", ""
                icon = str(i)
            st.markdown(f"""
            <div class='{cls}'>
                <div class='step-num {ncls}'>{icon}</div>
                <div>
                    <div class='step-t'>{title}</div>
                    <div class='step-s'>{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        if step > len(STEPS):
            st.markdown("""
            <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a; border-radius:10px; padding:16px 20px; margin-top:14px;'>
                <div style='font-size:13px; font-weight:600; color:#166534; margin-bottom:10px;'>Complaint successfully registered</div>
                <div style='background:white; border:1px solid #d1fae5; border-radius:8px; padding:14px 16px;'>
                    <div style='font-family:"JetBrains Mono",monospace; font-size:11px; color:#2563eb; font-weight:600; margin-bottom:8px;'>RES-2026-11482</div>
                    <div style='font-size:12.5px; color:#374151; line-height:2;'>
                        Category: <strong>Fraud / Unauthorized Transaction</strong><br>
                        Severity: <strong style='color:#dc2626;'>Critical</strong><br>
                        SLA Target: <strong>4 hours</strong><br>
                        Agent: <strong>Priya Desai — Fraud Investigation Unit</strong>
                    </div>
                    <div style='margin-top:10px;'>
                        <span style='background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700; padding:3px 12px; border-radius:10px;'>In Progress</span>
                    </div>
                </div>
                <div style='font-size:11.5px; color:#6b7280; margin-top:10px;'>WhatsApp acknowledgement sent to +91 98XXX XXXXX</div>
            </div>
            """, unsafe_allow_html=True)

        if step == 0:
            st.markdown("""
            <div class='chart-card' style='text-align:center; padding:40px 24px; margin-top:8px;'>
                <div style='font-size:13px; color:#9ca3af; font-weight:500;'>Click "Simulate Card Tap" to begin the journey</div>
                <div style='font-size:12.5px; color:#d1d5db; margin-top:6px;'>Then use "Next Step" to advance through each stage</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — WHATSAPP JOURNEY DEMO
# ═══════════════════════════════════════════════════════════════
elif page == "WhatsApp Journey Demo":

    # ── Init session state ──
    if "wa_msgs" not in st.session_state:
        st.session_state["wa_msgs"] = [
            {"role": "bot",
             "text": "Welcome to Resolve.AI, Union Bank of India's 24x7 Complaint Assistant.\n\nPlease describe your issue and I will register your complaint, classify it, and connect you with the right team immediately.",
             "time": "09:01"}
        ]
        st.session_state["wa_ticket"] = None

    left_c, right_c = st.columns([1, 1.05])

    with left_c:
        st.markdown("<div class='sec'>How It Works</div>", unsafe_allow_html=True)
        how_steps = [
            ("01", "Customer taps NFC card or opens the WhatsApp link",    "No app, no login, no form — works on any smartphone"),
            ("02", "Customer types complaint in plain language",            "Voice notes and images also supported in production"),
            ("03", "AI pipeline classifies, scores, and drafts response",  "End-to-end in under 3 seconds"),
            ("04", "Ticket created and agent assigned automatically",       "Critical complaints go directly to specialist teams"),
            ("05", "Customer receives live status updates via WhatsApp",    "No need to call or visit a branch"),
        ]
        for num, title, sub in how_steps:
            st.markdown(f"""
            <div style='display:flex; gap:16px; align-items:flex-start; padding:14px 0; border-bottom:1px solid #f3f4f6;'>
                <div style='font-size:20px; font-weight:800; color:#bfdbfe; font-family:"JetBrains Mono",monospace; min-width:30px; line-height:1;'>{num}</div>
                <div>
                    <div style='font-size:13.5px; font-weight:500; color:#111827;'>{title}</div>
                    <div style='font-size:12px; color:#9ca3af; margin-top:3px;'>{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
            <div style='font-size:12px; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:12px;'>Measured Impact</div>
        """, unsafe_allow_html=True)
        for lbl, val in [
            ("Complaint registered in",    "Under 5 seconds"),
            ("Resolution time improvement","50–60% faster"),
            ("Call centre load reduction", "40% reduction"),
            ("Manual effort eliminated",   "70%"),
        ]:
            st.markdown(f"""
            <div class='metric-row'>
                <span class='metric-label'>{lbl}</span>
                <span class='metric-val'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Sample complaint buttons — clicking sends the message directly
        st.markdown("<div class='sec'>Load a Sample Complaint</div>", unsafe_allow_html=True)
        WA_SAMPLES = [
            "My ATM swallowed my card and deducted Rs 10,000 but no cash came out.",
            "Unauthorised transfer of Rs 45,000 from my account at 3am. Please help immediately.",
            "Unable to login to net banking. Session timeout error for 3 days.",
            "EMI deducted twice this month. Please refund the extra amount.",
        ]
        for i, s in enumerate(WA_SAMPLES):
            btn_label = s[:50] + ("..." if len(s) > 50 else "")
            if st.button(btn_label, key=f"was{i}", use_container_width=True):
                now = datetime.now().strftime("%H:%M")
                st.session_state["wa_msgs"].append({"role": "user", "text": s, "time": now})
                res2   = engine.analyze(s, df)
                sla2   = SLA_MAP[res2["severity"]]
                nid2   = f"RES-2026-{random.randint(11000,12999)}"
                resp2  = generate_ai_response(
                    s, res2["category"], res2["severity"],
                    "Valued Customer", "UBOI*****", nid2, hf_key or None)
                ai_t2  = datetime.now().strftime("%H:%M")
                st.session_state["wa_msgs"].append({"role": "bot", "text": resp2, "time": ai_t2})
                st.session_state["wa_ticket"] = {
                    "id": nid2, "category": res2["category"],
                    "severity": res2["severity"], "sla": sla2, "time": ai_t2,
                }
                st.rerun()

    with right_c:
        st.markdown("<div class='sec'>Live WhatsApp Simulation</div>", unsafe_allow_html=True)

        # Build chat bubbles HTML
        chat_html = ""
        for m in st.session_state["wa_msgs"]:
            txt_esc = m["text"].replace("&","&amp;").replace("<","&lt;").replace("\n","<br>")
            if m["role"] == "bot":
                chat_html += f"<div class='wa-in'>{txt_esc}<div class='wa-t'>{m['time']}</div></div>"
            else:
                chat_html += f"<div class='wa-out'>{txt_esc}<div class='wa-t'>{m['time']}</div></div>"

        if st.session_state["wa_ticket"]:
            t = st.session_state["wa_ticket"]
            chat_html += f"""
            <div class='wa-ticket'>
                <div class='wa-ticket-header'>Resolve.AI — Complaint Registered</div>
                <div class='wa-ticket-row'><b>Ticket ID:</b> {t['id']}</div>
                <div class='wa-ticket-row'><b>Category:</b> {t['category']}</div>
                <div class='wa-ticket-row'><b>Severity:</b> <span class='wa-ticket-sev'>{t['severity']}</span></div>
                <div class='wa-ticket-row'><b>SLA Target:</b> {t['sla']} hours</div>
                <div class='wa-ticket-row'><b>Status:</b> Agent assigned</div>
                <div class='wa-status-pill'>In Progress</div>
                <div class='wa-t'>{t['time']}</div>
            </div>"""

        phone_html = f"""
        <div class='phone-wrap'>
            <div class='phone-outer'>
                <div class='phone-screen'>
                    <div class='wa-bar'>
                        <div class='wa-av'>R</div>
                        <div>
                            <div class='wa-n'>Resolve.AI Support</div>
                            <div class='wa-s'>Union Bank of India — 24x7 Active</div>
                        </div>
                    </div>
                    <div class='wa-chat'>{chat_html}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(phone_html, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        user_msg = st.text_input(
            "Type your complaint",
            placeholder="e.g. My debit card was blocked without notice…",
            key="wa_inp"
        )

        bc1, bc2 = st.columns([3, 1])
        send_btn  = bc1.button("Send Message", type="primary", use_container_width=True)
        reset_btn = bc2.button("Clear Chat",   use_container_width=True)

        if reset_btn:
            st.session_state["wa_msgs"]   = [
                {"role":"bot",
                 "text":"Welcome to Resolve.AI, Union Bank of India's 24x7 Complaint Assistant.\n\nPlease describe your issue and I will register your complaint, classify it, and connect you with the right team immediately.",
                 "time":"09:01"}
            ]
            st.session_state["wa_ticket"] = None
            st.rerun()

        if send_btn and user_msg.strip():
            now = datetime.now().strftime("%H:%M")
            st.session_state["wa_msgs"].append({"role":"user","text":user_msg,"time":now})

            with st.spinner("Resolve.AI is processing your message…"):
                res    = engine.analyze(user_msg, df)
                sla    = SLA_MAP[res["severity"]]
                new_id = f"RES-2026-{random.randint(11000,12999)}"
                resp   = generate_ai_response(
                    user_msg, res["category"], res["severity"],
                    "Valued Customer", "UBOI*****", new_id, hf_key or None)

            ai_time = datetime.now().strftime("%H:%M")
            st.session_state["wa_msgs"].append({"role":"bot","text":resp,"time":ai_time})
            st.session_state["wa_ticket"] = {
                "id":       new_id,
                "category": res["category"],
                "severity": res["severity"],
                "sla":      sla,
                "time":     ai_time,
            }
            st.rerun()

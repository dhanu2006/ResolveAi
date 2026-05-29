"""
app.py — Resolve.AI  |  Unified Complaint Intelligence Platform
Enterprise prototype — Union Bank of India deployment
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random, os, time

st.set_page_config(
    page_title="Resolve.AI — Complaint Intelligence",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
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
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
DESIGN = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    letter-spacing: -0.01em;
}
.main .block-container { padding: 0 2rem 2rem; max-width: 100%; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #08090d;
    border-right: 1px solid #1e2028;
}
section[data-testid="stSidebar"] * { color: #c8cad4 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 6px;
    transition: background 0.15s;
    display: block;
    font-size: 13.5px;
    font-weight: 400;
    color: #8890a4 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #13151e; color: #e2e4ee !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 11px; color: #464a5c !important; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
section[data-testid="stSidebar"] .stTextInput input { background: #13151e !important; border: 1px solid #1e2028 !important; color: #c8cad4 !important; font-size: 12px; }

/* ── Page header strip ── */
.page-header {
    background: #08090d;
    border-bottom: 1px solid #1a1c24;
    padding: 20px 32px 18px;
    margin: 0 -2rem 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.page-header-left { display: flex; align-items: center; gap: 16px; }
.page-header-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1e4db7, #0ea5e9);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 700; color: white; letter-spacing: -1px;
}
.page-header-title { font-size: 17px; font-weight: 600; color: #e8eaf4; line-height: 1; }
.page-header-sub   { font-size: 12px; color: #565a6e; margin-top: 3px; }
.page-header-bank  { font-size: 11px; color: #2a6fd4; background: #0d1a30; border: 1px solid #1a3460; padding: 4px 12px; border-radius: 20px; font-weight: 500; }

/* ── KPI Cards ── */
.kpi-card {
    background: #0e1018;
    border: 1px solid #1a1d27;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2a2e40; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-blue::before   { background: #2a6fd4; }
.kpi-red::before    { background: #dc2626; }
.kpi-amber::before  { background: #d97706; }
.kpi-rose::before   { background: #e11d48; }
.kpi-green::before  { background: #16a34a; }
.kpi-num   { font-size: 34px; font-weight: 600; color: #e8eaf4; line-height: 1; letter-spacing: -1.5px; font-family: 'DM Mono', monospace; }
.kpi-label { font-size: 12px; color: #565a6e; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.kpi-delta { font-size: 11px; margin-top: 8px; font-weight: 500; }
.delta-up   { color: #dc2626; }
.delta-down { color: #16a34a; }

/* ── Section header ── */
.sec-hdr {
    font-size: 13px;
    font-weight: 600;
    color: #565a6e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1d27;
}

/* ── Severity / Status pills ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.pill-critical { background: #1f0d0d; color: #f87171; border: 1px solid #3d1515; }
.pill-high     { background: #1e1308; color: #fb923c; border: 1px solid #3d2810; }
.pill-medium   { background: #1c1a08; color: #fbbf24; border: 1px solid #3d3610; }
.pill-low      { background: #0b1a10; color: #4ade80; border: 1px solid #144020; }
.pill-open       { background: #0d1525; color: #60a5fa; border: 1px solid #1a3060; }
.pill-progress   { background: #130f22; color: #a78bfa; border: 1px solid #2a1f44; }
.pill-escalated  { background: #1e1008; color: #fb923c; border: 1px solid #3d2210; }
.pill-resolved   { background: #0b1a10; color: #4ade80; border: 1px solid #144020; }

/* ── Complaint row card ── */
.complaint-row {
    background: #0e1018;
    border: 1px solid #1a1d27;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.15s, background 0.15s;
    cursor: pointer;
}
.complaint-row:hover { border-color: #2a2e40; background: #11131c; }
.complaint-row.sla-breach { border-left: 3px solid #dc2626; }
.c-id   { font-family: 'DM Mono', monospace; font-size: 11px; color: #2a6fd4; font-weight: 500; }
.c-text { font-size: 13.5px; color: #b0b4c4; margin: 4px 0; line-height: 1.5; }
.c-meta { font-size: 11px; color: #464a5c; }
.sla-warn { font-size: 11px; color: #dc2626; font-weight: 500; }

/* ── AI response box ── */
.ai-box {
    background: #090d18;
    border: 1px solid #1a2d4a;
    border-left: 3px solid #2a6fd4;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 13.5px;
    color: #b8bdd0;
    line-height: 1.75;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
}

/* ── NLP chips ── */
.nlp-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.nlp-chip {
    background: #0e1018;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 8px 16px;
    display: flex; flex-direction: column; align-items: center; gap: 2px;
}
.nlp-chip-label { font-size: 10px; color: #464a5c; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.nlp-chip-val   { font-size: 15px; font-weight: 600; color: #e2e4ee; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1a2d4a !important;
    border: 1px solid #2a4870 !important;
    color: #60a5fa !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 6px 18px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { background: #1e3560 !important; border-color: #3a5890 !important; }
.stButton > button[kind="primary"] {
    background: #1e4db7 !important;
    border-color: #2a5fd4 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover { background: #2a5fd4 !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: #0e1018 !important;
    border: 1px solid #1e2230 !important;
    color: #c8cad4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div { padding: 2px 8px !important; }
label, .stMarkdown p { color: #8890a4 !important; font-size: 13px !important; }
.stProgress > div > div { background: #1e4db7 !important; }
.stTabs [data-baseweb="tab-list"] { background: #0e1018; border-radius: 8px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #565a6e !important; font-family: 'DM Sans', sans-serif; font-size: 13px; padding: 7px 18px; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: #1a2d4a !important; color: #60a5fa !important; }
.stExpander { background: #0e1018; border: 1px solid #1a1d27 !important; border-radius: 10px; }
.stExpander summary { color: #8890a4 !important; font-size: 13.5px !important; }
div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; color: #e8eaf4 !important; }
.stAlert { background: #0e1018 !important; border-radius: 8px !important; }
h1,h2,h3 { color: #e8eaf4 !important; font-family: 'DM Sans', sans-serif !important; }
/* dark background for entire page */
.main { background: #06070b; }
section[data-testid="stSidebar"] > div { background: #08090d; }
/* plotly chart bg */
.js-plotly-plot { border-radius: 10px; }

/* ── NFC / WhatsApp demo specific ── */
.nfc-stage {
    background: #06070b;
    border: 1px solid #1a1d27;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.nfc-card-wrap {
    display: flex;
    justify-content: center;
    margin: 24px 0;
}
.mobile-frame {
    background: #1a1a2e;
    border-radius: 40px;
    padding: 12px;
    border: 2px solid #2a2a3e;
    box-shadow: 0 32px 80px rgba(0,0,0,0.8), 0 0 0 1px #3a3a5e;
    width: 320px;
    display: inline-block;
}
.mobile-screen {
    background: #e5ddd5;
    border-radius: 28px;
    overflow: hidden;
    min-height: 560px;
}
.wa-header {
    background: #075e54;
    padding: 12px 16px 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.wa-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: #128c7e;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700; color: white;
}
.wa-name  { font-size: 15px; font-weight: 600; color: white; }
.wa-sub   { font-size: 11px; color: rgba(255,255,255,0.7); }
.wa-body  { padding: 10px 10px; min-height: 440px; }
.wa-msg-in {
    background: white;
    border-radius: 0 12px 12px 12px;
    padding: 8px 12px;
    margin: 6px 40px 6px 4px;
    font-size: 12.5px;
    color: #1a1a1a;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    position: relative;
}
.wa-msg-out {
    background: #dcf8c6;
    border-radius: 12px 0 12px 12px;
    padding: 8px 12px;
    margin: 6px 4px 6px 40px;
    font-size: 12.5px;
    color: #1a1a1a;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    text-align: left;
}
.wa-time { font-size: 10px; color: #8a8a8a; text-align: right; margin-top: 2px; }
.wa-typing { font-size: 11px; color: #666; font-style: italic; padding: 4px 14px; }
.ticket-popup {
    background: #0a1628;
    border: 1px solid #1e3d6a;
    border-left: 3px solid #2a6fd4;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    text-align: left;
}
.ticket-popup .t-id   { font-family: 'DM Mono', monospace; font-size: 11px; color: #2a6fd4; font-weight: 600; }
.ticket-popup .t-row  { font-size: 12px; color: #8890a4; margin-top: 6px; }
.ticket-popup .t-val  { color: #c8cad4; font-weight: 500; }
.ticket-popup .t-status { font-size: 11px; background: #0b1a30; border: 1px solid #1a3460; color: #60a5fa; padding: 2px 10px; border-radius: 12px; display: inline-block; margin-top: 8px; }

/* Nav radio buttons look like menu items */
section[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }

/* heatmap / chart container */
.chart-wrap {
    background: #0e1018;
    border: 1px solid #1a1d27;
    border-radius: 12px;
    padding: 20px;
}
</style>
"""
st.markdown(DESIGN, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising NLP engine…")
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

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
SLA_MAP        = {"Critical": 4, "High": 24, "Medium": 48, "Low": 72}
SEV_COLORS     = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#22c55e"}
SENTIMENT_MAP  = {"Angry": "#ef4444", "Frustrated": "#f97316", "Concerned": "#eab308", "Neutral": "#6b7280"}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 8px 16px; border-bottom: 1px solid #1e2028; margin-bottom: 16px;'>
        <div style='font-size:20px; font-weight:700; color:#e8eaf4; letter-spacing:-0.5px;'>Resolve<span style='color:#2a6fd4;'>.</span>AI</div>
        <div style='font-size:11px; color:#464a5c; margin-top:3px;'>Complaint Intelligence Platform</div>
        <div style='font-size:10px; color:#2a6fd4; margin-top:8px; background:#0d1a30; border:1px solid #1a3460; padding:3px 10px; border-radius:12px; display:inline-block;'>Union Bank of India</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("NAVIGATION")
    page = st.radio("", [
        "Command Centre",
        "Live Complaint Intake",
        "Complaint Explorer",
        "Root Cause Intelligence",
        "NFC — Premium Channel",
        "WhatsApp Journey Demo",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("CONFIGURATION")
    hf_key = st.text_input("HuggingFace API Token", type="password", placeholder="hf_…", label_visibility="visible")
    if hf_key:
        st.markdown("<div style='font-size:11px; color:#16a34a; margin-top:4px;'>Enhanced AI active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:11px; color:#464a5c; margin-top:4px;'>Template responses active</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("SYSTEM STATUS")
    st.markdown("""
    <div style='background:#0b1a0d; border:1px solid #144020; border-radius:8px; padding:10px 12px;'>
        <div style='font-size:11px; color:#4ade80; display:flex; align-items:center; gap:6px;'>
            <span style='width:6px; height:6px; background:#4ade80; border-radius:50%; display:inline-block;'></span>
            NLP Engine — Online
        </div>
        <div style='font-size:11px; color:#4ade80; display:flex; align-items:center; gap:6px; margin-top:6px;'>
            <span style='width:6px; height:6px; background:#4ade80; border-radius:50%; display:inline-block;'></span>
            Database — Connected
        </div>
        <div style='font-size:11px; color:#fbbf24; display:flex; align-items:center; gap:6px; margin-top:6px;'>
            <span style='width:6px; height:6px; background:#fbbf24; border-radius:50%; display:inline-block;'></span>
            WhatsApp API — Simulated
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SHARED HEADER COMPONENT
# ─────────────────────────────────────────────────────────────
PAGE_TITLES = {
    "Command Centre":          ("Command Centre",           "Real-time operations overview — complaint volume, SLA status, and team workload"),
    "Live Complaint Intake":   ("Live Complaint Intake",    "Ingest complaints from any channel with AI-powered classification and response drafting"),
    "Complaint Explorer":      ("Complaint Explorer",       "360-degree view of every complaint — filter, investigate, and action from one place"),
    "Root Cause Intelligence": ("Root Cause Intelligence",  "Detect systemic patterns, predict complaint spikes, and identify operational failures"),
    "NFC — Premium Channel":   ("NFC Premium Channel",      "One-tap complaint registration for priority customers via NFC SmartCard"),
    "WhatsApp Journey Demo":   ("WhatsApp Journey Demo",    "Full end-to-end simulation of the customer complaint journey via WhatsApp"),
}
title, subtitle = PAGE_TITLES[page]
st.markdown(f"""
<div class='page-header'>
    <div class='page-header-left'>
        <div class='page-header-logo'>R</div>
        <div>
            <div class='page-header-title'>{title}</div>
            <div class='page-header-sub'>{subtitle}</div>
        </div>
    </div>
    <div class='page-header-bank'>Union Bank of India — Production Prototype</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def sev_pill(s):
    return f"<span class='pill pill-{s.lower()}'>{s}</span>"

def status_pill(s):
    cls = s.lower().replace(" ", "-").replace("in-progress","progress")
    return f"<span class='pill pill-{cls}'>{s}</span>"

def plotly_dark_layout(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8890a4", size=12),
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8890a4"),
        xaxis=dict(gridcolor="#1a1d27", linecolor="#1a1d27", tickcolor="#1a1d27"),
        yaxis=dict(gridcolor="#1a1d27", linecolor="#1a1d27", tickcolor="#1a1d27"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# PAGE 1  COMMAND CENTRE
# ═══════════════════════════════════════════════════════════════
if page == "Command Centre":
    total     = len(df)
    open_c    = len(df[df["status"] == "Open"])
    escalated = len(df[df["status"] == "Escalated"])
    breached  = len(df[df["sla_breached"] == True])
    resolved  = len(df[df["status"] == "Resolved"])
    fraud_c   = len(df[df["category"] == "Fraud / Unauthorized Transaction"])

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (k1, str(total),    "Total Complaints",  "kpi-blue",  "+12 today", True),
        (k2, str(open_c),   "Open",              "kpi-red",   "Requires action", True),
        (k3, str(escalated),"Escalated",         "kpi-amber", f"{int(escalated/total*100)}% of open", True),
        (k4, str(breached), "SLA Breached",      "kpi-rose",  "Immediate review required", True),
        (k5, str(resolved), "Resolved",          "kpi-green", f"{int(resolved/total*100)}% resolution rate", False),
        (k6, str(fraud_c),  "Fraud Alerts",      "kpi-red",   "Critical priority", True),
    ]
    for col, num, label, klass, delta, is_bad in kpis:
        color = "#ef4444" if is_bad else "#4ade80"
        with col:
            st.markdown(f"""
            <div class='kpi-card {klass}'>
                <div class='kpi-num'>{num}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-delta' style='color:{color};'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    # Charts row 1
    st.markdown("<div class='sec-hdr'>Volume &amp; Distribution</div>", unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns([1.4, 1, 1])

    with ch1:
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["ShortName"] = cat_counts["Category"].str.split("/").str[0].str.strip()
        fig = px.bar(cat_counts, x="Count", y="ShortName", orientation="h",
                     color="Count", color_continuous_scale=[[0,"#1a2d4a"],[1,"#2a6fd4"]],
                     text="Count")
        fig.update_traces(textposition="outside", textfont_color="#8890a4", textfont_size=11, marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig = plotly_dark_layout(fig, 280)
        fig.update_yaxes(title="")
        fig.update_xaxes(title="")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("Complaints by category")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        sev = df["severity"].value_counts()
        fig2 = px.pie(values=sev.values, names=sev.index,
                      color=sev.index, color_discrete_map=SEV_COLORS, hole=0.55)
        fig2.update_traces(textposition="outside", textinfo="percent+label",
                           marker_line_color="#06070b", marker_line_width=2)
        fig2 = plotly_dark_layout(fig2, 280)
        fig2.update_layout(showlegend=False)
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("Severity distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch3:
        ch_data = df.groupby("channel")["complaint_id"].count().reset_index()
        ch_data.columns = ["Channel","Count"]
        ch_colors_map = {"WhatsApp": "#25D366", "Email": "#2a6fd4", "Web Form": "#f97316"}
        fig3 = px.bar(ch_data, x="Channel", y="Count",
                      color="Channel", color_discrete_map=ch_colors_map, text="Count")
        fig3.update_traces(textposition="outside", textfont_color="#8890a4", textfont_size=11, marker_line_width=0)
        fig3 = plotly_dark_layout(fig3, 280)
        fig3.update_layout(showlegend=False)
        fig3.update_xaxes(title="")
        fig3.update_yaxes(title="")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("Complaints by intake channel")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts row 2
    ch4, ch5 = st.columns([1.6, 1])

    with ch4:
        st.markdown("<div class='sec-hdr'>Complaint Volume Trend</div>", unsafe_allow_html=True)
        df_tmp = df.copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["created_at"]).dt.date
        trend = df_tmp.groupby(["date","category"])["complaint_id"].count().reset_index()
        trend.columns = ["Date","Category","Count"]
        total_trend = df_tmp.groupby("date")["complaint_id"].count().reset_index()
        total_trend.columns = ["Date","Count"]
        fig4 = px.area(total_trend, x="Date", y="Count",
                       color_discrete_sequence=["#2a6fd4"])
        fig4.update_traces(fillcolor="rgba(42,111,212,0.12)", line_color="#2a6fd4", line_width=2)
        fig4 = plotly_dark_layout(fig4, 240)
        fig4.update_xaxes(title="")
        fig4.update_yaxes(title="")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch5:
        st.markdown("<div class='sec-hdr'>Sentiment Breakdown</div>", unsafe_allow_html=True)
        sent = df["sentiment"].value_counts().reset_index()
        sent.columns = ["Sentiment","Count"]
        fig5 = px.bar(sent, x="Sentiment", y="Count",
                      color="Sentiment", color_discrete_map=SENTIMENT_MAP, text="Count")
        fig5.update_traces(textposition="outside", textfont_color="#8890a4", textfont_size=11, marker_line_width=0)
        fig5 = plotly_dark_layout(fig5, 240)
        fig5.update_layout(showlegend=False)
        fig5.update_xaxes(title="")
        fig5.update_yaxes(title="")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Priority queue
    st.markdown("<div class='sec-hdr'>Priority Queue — Requires Attention</div>", unsafe_allow_html=True)
    urgent = df[df["status"].isin(["Open","Escalated"])].copy()
    urgent["rank"] = urgent["severity"].map(SEVERITY_ORDER)
    urgent = urgent.sort_values(["rank","created_at"]).head(8)
    for _, row in urgent.iterrows():
        breach_cls = "sla-breach" if row["sla_breached"] else ""
        breach_tag = "<span class='sla-warn'>SLA BREACH</span>" if row["sla_breached"] else ""
        st.markdown(f"""
        <div class='complaint-row {breach_cls}'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div style='flex:1;'>
                    <span class='c-id'>{row['complaint_id']}</span>
                    <span style='font-size:11px; color:#464a5c; margin-left:10px;'>{row['channel']}  ·  {row['created_at'][:16]}</span>
                    <div class='c-text'>{row['complaint_text'][:110]}…</div>
                    <div class='c-meta'>{row['customer_name']}  ·  {row['branch']}  ·  SLA: {row['sla_hours']}h</div>
                </div>
                <div style='text-align:right; flex-shrink:0; margin-left:16px;'>
                    {sev_pill(row['severity'])}
                    <div style='margin-top:6px;'>{status_pill(row['status'])}</div>
                    <div style='margin-top:6px;'>{breach_tag}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2  LIVE COMPLAINT INTAKE
# ═══════════════════════════════════════════════════════════════
elif page == "Live Complaint Intake":
    col_form, col_result = st.columns([1, 1.15])

    SAMPLES = [
        ("ATM — cash not dispensed",
         "My ATM card was swallowed by the machine at Bandra branch and Rs 10,000 was debited from my account but cash was not dispensed. Transaction reference: UBI2026049821."),
        ("Fraud — unauthorised transfer",
         "An unauthorised transfer of Rs 45,000 was made from my savings account at 2:47am. I did not authorise this. I received no OTP. Please freeze the account and investigate immediately."),
        ("Net banking — login failure",
         "I have been unable to login to mobile banking for 4 days. The app shows session timeout immediately after OTP verification. I need to make urgent payments."),
        ("Loan — EMI debited twice",
         "My home loan EMI of Rs 28,500 was debited twice on the same day this month. This has caused my account balance to go negative and incurred overdraft charges."),
        ("Branch — poor service",
         "The staff at Nagpur Central branch refused to assist my elderly mother with a simple demand draft request and asked her to come back the next day without any reason."),
        ("KYC — account frozen",
         "My account has been frozen without prior notice or communication. I submitted updated KYC documents at the branch three weeks ago. I urgently need access to my funds."),
    ]

    with col_form:
        st.markdown("<div class='sec-hdr'>Complaint Details</div>", unsafe_allow_html=True)
        channel      = st.selectbox("Intake Channel", ["WhatsApp", "Email", "Web Form", "NFC SmartCard", "Branch"])
        customer_name= st.text_input("Customer Name", value="Rahul Sharma")
        account_num  = st.text_input("Account Number", value="UBOI123456789")
        complaint_txt= st.text_area("Complaint Description", height=130, placeholder="Paste or type the customer complaint text…")

        st.markdown("<div class='sec-hdr'>Sample Complaints</div>", unsafe_allow_html=True)
        for i, (label, sample) in enumerate(SAMPLES):
            if st.button(label, key=f"s{i}", use_container_width=True):
                st.session_state["_sample"] = sample
        if "_sample" in st.session_state and not complaint_txt.strip():
            complaint_txt = st.session_state["_sample"]

        go_btn = st.button("Analyse and Register Complaint", type="primary", use_container_width=True)

    with col_result:
        st.markdown("<div class='sec-hdr'>AI Analysis Output</div>", unsafe_allow_html=True)
        if go_btn and complaint_txt.strip():
            with st.spinner("Running NLP pipeline…"):
                res = engine.analyze(complaint_txt, df)
                new_id = f"RES-2026-{random.randint(11000,12999)}"
                sla = SLA_MAP[res["severity"]]

            st.success(f"Complaint registered — {new_id}")

            # NLP chips
            conf_pct = int(res["confidence"] * 100)
            st.markdown(f"""
            <div class='nlp-row'>
                <div class='nlp-chip'><span class='nlp-chip-label'>Category</span><span class='nlp-chip-val'>{res['category'].split('/')[0].strip()}</span></div>
                <div class='nlp-chip'><span class='nlp-chip-label'>Severity</span><span class='nlp-chip-val' style='color:{SEV_COLORS[res["severity"]]};'>{res['severity']}</span></div>
                <div class='nlp-chip'><span class='nlp-chip-label'>Sentiment</span><span class='nlp-chip-val'>{res['sentiment']}</span></div>
                <div class='nlp-chip'><span class='nlp-chip-label'>Confidence</span><span class='nlp-chip-val'>{conf_pct}%</span></div>
                <div class='nlp-chip'><span class='nlp-chip-label'>SLA Target</span><span class='nlp-chip-val'>{sla}h</span></div>
            </div>""", unsafe_allow_html=True)

            st.progress(res["confidence"])
            if res["alternatives"]:
                alts = "  ·  ".join(f"{a['category'].split('/')[0].strip()} ({int(a['confidence']*100)}%)" for a in res["alternatives"])
                st.caption(f"Other candidate categories: {alts}")

            # Duplicate detection
            if res["is_duplicate"]:
                st.warning(f"{len(res['duplicates'])} similar complaint(s) detected — possible systemic issue")
                for dup in res["duplicates"]:
                    st.markdown(f"- `{dup['complaint_id']}` ({dup['status']}) · {int(dup['similarity']*100)}% match · {dup['text_snippet']}")
            else:
                st.markdown("<div style='font-size:12px; color:#4ade80; margin:8px 0;'>No duplicate complaints found</div>", unsafe_allow_html=True)

            # AI response
            st.markdown("<div class='sec-hdr'>Drafted Customer Response</div>", unsafe_allow_html=True)
            with st.spinner("Generating AI response…"):
                response = generate_ai_response(
                    complaint_txt, res["category"], res["severity"],
                    customer_name, account_num, new_id,
                    hf_key if hf_key else None,
                )
            st.markdown(f'<div class="ai-box">{response}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.button("Send via WhatsApp",   use_container_width=True)
            c2.button("Assign to Agent",     use_container_width=True)
            c3.button("Escalate to Manager", use_container_width=True)

        elif go_btn:
            st.warning("Please enter or select a complaint description.")
        else:
            st.markdown("""
            <div style='background:#0e1018; border:1px solid #1a1d27; border-radius:12px; padding:48px 32px; text-align:center; margin-top:12px;'>
                <div style='font-size:13px; color:#464a5c; line-height:1.7;'>
                    Fill in the complaint details on the left, then click<br>
                    <strong style='color:#8890a4;'>Analyse and Register Complaint</strong><br><br>
                    The NLP pipeline will classify the complaint, score severity and sentiment,<br>
                    check for duplicates, and draft an AI-powered customer response.
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3  COMPLAINT EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "Complaint Explorer":
    f1, f2, f3, f4 = st.columns(4)
    sel_ch  = f1.multiselect("Channel",   df["channel"].unique(),   default=list(df["channel"].unique()))
    sel_sev = f2.multiselect("Severity",  ["Critical","High","Medium","Low"], default=["Critical","High","Medium","Low"])
    sel_sta = f3.multiselect("Status",    df["status"].unique(),    default=list(df["status"].unique()))
    sel_cat = f4.multiselect("Category",  df["category"].unique(),  default=list(df["category"].unique()))

    filt = df[df["channel"].isin(sel_ch) & df["severity"].isin(sel_sev) &
              df["status"].isin(sel_sta) & df["category"].isin(sel_cat)]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Filtered", len(filt))
    m2.metric("SLA Breaches", int(filt["sla_breached"].sum()))
    avg_res = filt["resolution_time_hrs"].dropna().mean()
    m3.metric("Avg Resolution", f"{avg_res:.0f}h" if not np.isnan(avg_res) else "—")
    m4.metric("Critical Open", len(filt[(filt["severity"]=="Critical") & (filt["status"].isin(["Open","Escalated"]))]))

    st.markdown(f"<div class='sec-hdr'>Showing {len(filt)} complaints</div>", unsafe_allow_html=True)

    for _, row in filt.sort_values("created_at", ascending=False).head(30).iterrows():
        with st.expander(f"{row['complaint_id']}  —  {row['category']}  —  {row['customer_name']}"):
            d1, d2 = st.columns([1.6, 1])
            with d1:
                st.markdown(f"**Complaint**")
                st.markdown(f"<div class='c-text' style='font-size:14px;'>{row['complaint_text']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Customer:** {row['customer_name']}  &nbsp;·&nbsp;  **Account:** `{row['account_number']}`  &nbsp;·&nbsp;  **Branch:** {row['branch']}", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <div style='background:#0e1018; border:1px solid #1a1d27; border-radius:10px; padding:14px 16px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                        {sev_pill(row['severity'])} {status_pill(row['status'])}
                    </div>
                    <div class='c-meta' style='line-height:2;'>
                        Channel: {row['channel']}<br>
                        Created: {row['created_at'][:16]}<br>
                        Sentiment: {row['sentiment']}<br>
                        SLA: {row['sla_hours']}h {'<span class="sla-warn">BREACHED</span>' if row['sla_breached'] else "<span style='color:#4ade80;'>On track</span>"}<br>
                        Agent: {row['agent_assigned'] or 'Unassigned'}
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button("Generate AI Response", key=f"ai_{row['complaint_id']}", use_container_width=True):
                    with st.spinner("Drafting response…"):
                        resp = generate_ai_response(
                            row["complaint_text"], row["category"], row["severity"],
                            row["customer_name"], row["account_number"],
                            row["complaint_id"], hf_key if hf_key else None,
                        )
                    st.markdown(f'<div class="ai-box" style="margin-top:12px;">{resp}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4  ROOT CAUSE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif page == "Root Cause Intelligence":
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Recurring Patterns", "SLA Heatmap"])

    with tab1:
        df_t = df.copy()
        df_t["date"] = pd.to_datetime(df_t["created_at"]).dt.date
        trend = df_t.groupby(["date","category"])["complaint_id"].count().reset_index()
        trend.columns = ["Date","Category","Count"]
        trend["ShortCat"] = trend["Category"].str.split("/").str[0].str.strip()
        palette = ["#2a6fd4","#ef4444","#f97316","#eab308","#22c55e","#a78bfa"]
        fig = px.line(trend, x="Date", y="Count", color="ShortCat",
                      color_discrete_sequence=palette)
        fig.update_traces(line_width=2)
        fig = plotly_dark_layout(fig, 340)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Complaints")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("Daily complaint volume by category")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        daily = df_t.groupby("date")["complaint_id"].count()
        mu, sigma = daily.mean(), daily.std()
        spikes = daily[daily > mu + 1.5 * sigma]
        if not spikes.empty:
            st.markdown(f"""
            <div style='background:#1f0d0d; border:1px solid #3d1515; border-radius:10px; padding:14px 20px; margin-top:12px;'>
                <div style='font-size:13px; font-weight:600; color:#f87171;'>Complaint spike detected on {len(spikes)} day(s)</div>
                <div style='font-size:12px; color:#8890a4; margin-top:4px;'>Dates: {", ".join(str(d) for d in spikes.index)}  ·  Threshold: {mu + 1.5*sigma:.1f} complaints/day</div>
                <div style='font-size:12px; color:#8890a4; margin-top:4px;'>Recommend: activate surge-response protocol and cross-reference with known service incidents</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        rec = df.groupby(["category","severity"])["complaint_id"].count().reset_index()
        rec.columns = ["Category","Severity","Count"]
        rec["ShortCat"] = rec["Category"].str.split("/").str[0].str.strip()
        fig2 = px.bar(rec, x="ShortCat", y="Count", color="Severity",
                      color_discrete_map=SEV_COLORS, barmode="stack",
                      text_auto=True)
        fig2.update_traces(textfont_color="rgba(0,0,0,0.6)", textfont_size=10, marker_line_width=0)
        fig2 = plotly_dark_layout(fig2, 320)
        fig2.update_xaxes(title="", tickangle=-15)
        fig2.update_yaxes(title="Complaints")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("Complaint volume by category and severity")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Systemic Issues Detected</div>", unsafe_allow_html=True)
        top = df[df["severity"].isin(["Critical","High"])].groupby("category")["complaint_id"].count().sort_values(ascending=False)
        for cat, cnt in top.items():
            pct = int(cnt / len(df) * 100)
            short = cat.split("/")[0].strip()
            st.markdown(f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:10px;'><div style='font-size:13px; color:#c8cad4; width:200px;'>{short}</div><div style='flex:1; background:#1a1d27; border-radius:4px; height:6px;'><div style='background:#2a6fd4; width:{pct}%; height:6px; border-radius:4px;'></div></div><div style='font-size:12px; color:#565a6e; width:80px; text-align:right;'>{cnt} high/critical</div></div>", unsafe_allow_html=True)

    with tab3:
        heat = df.groupby(["category","severity"])["sla_breached"].mean().reset_index()
        heat["sla_breached"] = (heat["sla_breached"] * 100).round(1)
        heat_pivot = heat.pivot(index="category", columns="severity", values="sla_breached").fillna(0)
        for c in ["Critical","High","Medium","Low"]:
            if c not in heat_pivot.columns:
                heat_pivot[c] = 0
        heat_pivot = heat_pivot[["Critical","High","Medium","Low"]]
        heat_pivot.index = [i.split("/")[0].strip() for i in heat_pivot.index]

        fig3 = px.imshow(heat_pivot, color_continuous_scale=[[0,"#0e1018"],[0.5,"#1a2d4a"],[1,"#dc2626"]],
                         aspect="auto", text_auto=".1f")
        fig3.update_traces(textfont_color="white", textfont_size=12)
        fig3 = plotly_dark_layout(fig3, 320)
        fig3.update_coloraxes(colorbar_tickfont_color="#8890a4")
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.caption("SLA breach rate (%) by category and severity")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Darker red = higher % of complaints in that segment that breached their SLA deadline")

    # Insight summary
    breach_pct = len(df[df["sla_breached"]==True]) / len(df) * 100
    fraud_pct  = len(df[df["category"]=="Fraud / Unauthorized Transaction"]) / len(df) * 100
    st.markdown(f"""
    <div style='background:#090d18; border:1px solid #1a2d4a; border-left:3px solid #2a6fd4; border-radius:10px; padding:20px 24px; margin-top:20px;'>
        <div style='font-size:13px; font-weight:600; color:#60a5fa; margin-bottom:10px;'>AI Insight Summary</div>
        <div style='font-size:13px; color:#8890a4; line-height:2;'>
            Overall SLA breach rate is <strong style='color:#e2e4ee;'>{breach_pct:.1f}%</strong> — correlates with staffing gaps during 9–11am and 6–8pm peaks.<br>
            Fraud complaints represent <strong style='color:#e2e4ee;'>{fraud_pct:.1f}%</strong> of volume but carry the highest severity and regulatory exposure.<br>
            Recommendation: implement auto-escalation for any Critical complaint unresolved after 2 hours, and deploy a dedicated NLP triage bot for ATM and Net Banking complaints which account for the highest recurring volume.
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5  NFC PREMIUM CHANNEL
# ═══════════════════════════════════════════════════════════════
elif page == "NFC — Premium Channel":

    NFC_CSS = """
    <style>
    @keyframes nfc-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(212,175,55,0.5); }
        70%  { box-shadow: 0 0 0 24px rgba(212,175,55,0); }
        100% { box-shadow: 0 0 0 0 rgba(212,175,55,0); }
    }
    @keyframes tap-ripple {
        0%   { transform: scale(0.8); opacity:1; }
        100% { transform: scale(2.4); opacity:0; }
    }
    @keyframes phone-appear {
        0%   { opacity:0; transform: translateY(30px) scale(0.95); }
        100% { opacity:1; transform: translateY(0) scale(1); }
    }
    @keyframes msg-in {
        0%   { opacity:0; transform: translateX(-12px); }
        100% { opacity:1; transform: translateX(0); }
    }
    @keyframes step-glow {
        0%,100% { border-color: #1a3460; }
        50%      { border-color: #2a6fd4; box-shadow: 0 0 12px rgba(42,111,212,0.3); }
    }
    .nfc-card-outer {
        width: 380px; height: 240px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1a140a 0%, #2e2410 30%, #4a3a1a 55%, #2e2410 80%, #1a140a 100%);
        position: relative;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 20px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(212,175,55,0.2), inset 0 1px 0 rgba(212,175,55,0.3);
        overflow: hidden;
    }
    .nfc-card-outer:hover {
        transform: translateY(-4px) rotateX(3deg);
        box-shadow: 0 32px 80px rgba(0,0,0,0.8), 0 0 0 1px rgba(212,175,55,0.4), 0 0 40px rgba(212,175,55,0.1);
    }
    .nfc-card-shine {
        position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, transparent 50%, rgba(255,255,255,0.03) 100%);
        border-radius: 18px;
        pointer-events: none;
    }
    .nfc-card-lines {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(212,175,55,0.03) 2px, rgba(212,175,55,0.03) 4px);
        border-radius: 18px;
        pointer-events: none;
    }
    .nfc-card-chip {
        position: absolute; top: 70px; left: 28px;
        width: 44px; height: 34px;
        background: linear-gradient(135deg, #c8a84b, #f0d060, #c8a84b);
        border-radius: 6px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 4px rgba(0,0,0,0.4);
    }
    .nfc-card-chip::after {
        content: '';
        position: absolute;
        inset: 4px;
        border: 1px solid rgba(0,0,0,0.3);
        border-radius: 3px;
    }
    .nfc-symbol {
        position: absolute; top: 66px; right: 28px;
        display: flex; flex-direction: column; align-items: center; gap: 3px;
    }
    .nfc-arc {
        border: 2px solid;
        border-color: transparent transparent transparent rgba(212,175,55,0.7);
        border-radius: 50%;
    }
    .nfc-card-bank {
        position: absolute; bottom: 22px; left: 28px;
        font-size: 11px; font-weight: 600; color: rgba(212,175,55,0.8);
        letter-spacing: 0.12em; text-transform: uppercase; font-family: 'DM Mono', monospace;
    }
    .nfc-card-name {
        position: absolute; bottom: 42px; left: 28px;
        font-size: 14px; font-weight: 500; color: rgba(255,255,255,0.85);
        letter-spacing: 0.04em; font-family: 'DM Sans', sans-serif;
    }
    .nfc-card-num {
        position: absolute; bottom: 22px; right: 28px;
        font-size: 11px; color: rgba(212,175,55,0.6);
        font-family: 'DM Mono', monospace; letter-spacing: 0.1em;
    }
    .resolve-badge {
        position: absolute; top: 20px; right: 22px;
        font-size: 14px; font-weight: 700; color: #d4af37;
        letter-spacing: -0.5px;
    }
    .resolve-badge span { color: rgba(212,175,55,0.5); }
    .priority-badge {
        position: absolute; top: 20px; left: 28px;
        font-size: 9px; font-weight: 700; color: rgba(212,175,55,0.7);
        text-transform: uppercase; letter-spacing: 0.15em;
        border: 1px solid rgba(212,175,55,0.3); padding: 2px 8px; border-radius: 10px;
    }
    .tap-zone {
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,0,0,0);
        z-index: 10;
    }
    .phone-after {
        animation: phone-appear 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;
    }
    .step-card {
        background: #0e1018;
        border: 1px solid #1a1d27;
        border-radius: 10px;
        padding: 14px 16px;
        display: flex; align-items: flex-start; gap: 12px;
        margin-bottom: 8px;
        transition: all 0.3s;
    }
    .step-active {
        border-color: #2a6fd4;
        background: #090d18;
        animation: step-glow 2s ease-in-out infinite;
    }
    .step-done { border-color: #144020; background: #0b1a0d; }
    .step-num {
        width: 26px; height: 26px; border-radius: 50%;
        background: #1a1d27; display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600; color: #565a6e; flex-shrink: 0;
    }
    .step-num-done  { background: #144020; color: #4ade80; }
    .step-num-active{ background: #1a3460; color: #60a5fa; }
    .step-title { font-size: 13px; font-weight: 500; color: #c8cad4; }
    .step-sub   { font-size: 11px; color: #565a6e; margin-top: 2px; }
    </style>
    """
    st.markdown(NFC_CSS, unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        # Description
        st.markdown("""
        <div style='margin-bottom:24px;'>
            <div style='font-size:22px; font-weight:600; color:#d4af37; letter-spacing:-0.5px;'>Priority SmartCard</div>
            <div style='font-size:14px; color:#8890a4; margin-top:6px; line-height:1.7;'>
                One tap. Zero friction. Complaint registered in under 5 seconds.<br>
                Issued to premium and high-value customers of Union Bank of India.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Golden NFC card
        st.markdown("""
        <div style='display:flex; justify-content:center; margin: 8px 0 32px;'>
            <div class='nfc-card-outer' id='nfc-card'>
                <div class='nfc-card-shine'></div>
                <div class='nfc-card-lines'></div>
                <div class='priority-badge'>Priority</div>
                <div class='resolve-badge'>Resolve<span>.</span>AI</div>
                <div class='nfc-card-chip'></div>
                <div class='nfc-symbol'>
                    <div class='nfc-arc' style='width:28px;height:28px;'></div>
                    <div class='nfc-arc' style='width:20px;height:20px; margin-top:-20px;'></div>
                    <div class='nfc-arc' style='width:12px;height:12px; margin-top:-20px;'></div>
                </div>
                <div class='nfc-card-name'>RAHUL SHARMA</div>
                <div class='nfc-card-bank'>Union Bank of India</div>
                <div class='nfc-card-num'>**** 4821</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature specs
        specs = [
            ("NFC Chip", "NTAG213/215 — works with any NFC-enabled smartphone"),
            ("Response time", "Complaint registered in under 5 seconds"),
            ("No app required", "Opens WhatsApp directly — no login, no download"),
            ("Priority routing", "NFC complaints automatically queue at Critical priority"),
            ("Issued to", "Premium, Priority, and High-Value Account customers"),
        ]
        for label, desc in specs:
            st.markdown(f"""
            <div style='display:flex; gap:14px; align-items:flex-start; padding:10px 0; border-bottom:1px solid #1a1d27;'>
                <div style='font-size:12px; font-weight:600; color:#d4af37; min-width:120px;'>{label}</div>
                <div style='font-size:12px; color:#8890a4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        tap_btn = st.button("Simulate Card Tap", type="primary", use_container_width=True)

    with right:
        st.markdown("<div style='sec-hdr' style='font-size:13px; font-weight:600; color:#565a6e; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:16px;'>Journey Walkthrough</div>", unsafe_allow_html=True)

        step_state = st.session_state.get("nfc_step", 0)
        if tap_btn:
            step_state = 1
            st.session_state["nfc_step"] = 1

        STEPS = [
            ("Card detected", "NFC reader detects NTAG215 chip — customer account retrieved"),
            ("Identity verified", "Account number and mobile number matched — OTP pre-authorised"),
            ("WhatsApp opens", "Deep-link triggers WhatsApp with pre-filled customer context"),
            ("Complaint submitted", "Customer describes issue — message sent to Resolve.AI bot"),
            ("AI classification", "NLP pipeline runs: category Critical, sentiment Angry detected"),
            ("Agent notified", "Complaint assigned to Fraud team — SLA clock started at 4 hours"),
            ("Customer notified", "Acknowledgement and ticket ID sent back via WhatsApp"),
        ]

        for i, (title, sub) in enumerate(STEPS, 1):
            if step_state == 0:
                cls_card = "step-card"
                cls_num  = "step-num"
            elif i < step_state:
                cls_card = "step-card step-done"
                cls_num  = "step-num step-num-done"
            elif i == step_state:
                cls_card = "step-card step-active"
                cls_num  = "step-num step-num-active"
            else:
                cls_card = "step-card"
                cls_num  = "step-num"
            tick = "✓" if (step_state > 0 and i < step_state) else str(i)
            st.markdown(f"""
            <div class='{cls_card}'>
                <div class='{cls_num}'>{tick}</div>
                <div>
                    <div class='step-title'>{title}</div>
                    <div class='step-sub'>{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        if step_state > 0:
            # Auto-advance steps
            if step_state < len(STEPS) + 1:
                import time
                progress_placeholder = st.empty()
                for s in range(step_state, len(STEPS) + 1):
                    st.session_state["nfc_step"] = s
                    time.sleep(0.6)
                    st.rerun()

            # Show WhatsApp preview after full flow
            st.markdown("""
            <div style='background:#0b1a0d; border:1px solid #144020; border-radius:10px; padding:16px 20px; margin-top:16px;'>
                <div style='font-size:12px; font-weight:600; color:#4ade80; margin-bottom:10px;'>Complaint successfully registered</div>
                <div class='ticket-popup'>
                    <div class='t-id'>RES-2026-11482</div>
                    <div class='t-row'>Category: <span class='t-val'>Fraud / Unauthorized Transaction</span></div>
                    <div class='t-row'>Severity: <span class='t-val' style='color:#ef4444;'>Critical</span></div>
                    <div class='t-row'>SLA Target: <span class='t-val'>4 hours</span></div>
                    <div class='t-row'>Agent: <span class='t-val'>Priya Desai — Fraud Investigation Unit</span></div>
                    <div class='t-status'>In Progress</div>
                </div>
                <div style='font-size:11px; color:#565a6e; margin-top:10px;'>WhatsApp acknowledgement sent to +91 98XXX XXXXX</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Reset Demo", use_container_width=True):
                st.session_state["nfc_step"] = 0
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 6  WHATSAPP JOURNEY DEMO
# ═══════════════════════════════════════════════════════════════
elif page == "WhatsApp Journey Demo":

    WA_CSS = """
    <style>
    @keyframes msg-slide {
        0%   { opacity:0; transform:translateY(8px); }
        100% { opacity:1; transform:translateY(0); }
    }
    .msg-anim { animation: msg-slide 0.3s ease forwards; }
    </style>
    """
    st.markdown(WA_CSS, unsafe_allow_html=True)

    if "wa_messages" not in st.session_state:
        st.session_state["wa_messages"] = [
            {"role": "bot", "text": "Welcome to Resolve.AI — Union Bank of India's 24x7 Complaint Assistant.\n\nPlease describe your issue and I will register your complaint, classify it, and connect you with the right team immediately.", "time": "09:01"},
        ]
        st.session_state["wa_stage"] = "idle"
        st.session_state["wa_ticket"] = None

    left_col, right_col = st.columns([1, 1.05])

    with left_col:
        st.markdown("<div class='sec-hdr'>How it works</div>", unsafe_allow_html=True)
        steps_info = [
            ("01", "Customer taps NFC card or opens WhatsApp link", "No app, no login, no form"),
            ("02", "Customer types their complaint in plain language", "Voice notes and images supported in production"),
            ("03", "AI pipeline classifies, scores, and drafts response", "Under 3 seconds end-to-end"),
            ("04", "Ticket created and agent assigned automatically", "Critical complaints — dedicated fraud team"),
            ("05", "Customer receives real-time status updates", "No need to call or visit branch"),
        ]
        for num, title, sub in steps_info:
            st.markdown(f"""
            <div style='display:flex; gap:14px; align-items:flex-start; padding:14px 0; border-bottom:1px solid #1a1d27;'>
                <div style='font-size:22px; font-weight:700; color:#1a3460; font-family:"DM Mono",monospace; min-width:32px;'>{num}</div>
                <div>
                    <div style='font-size:13.5px; font-weight:500; color:#c8cad4;'>{title}</div>
                    <div style='font-size:12px; color:#565a6e; margin-top:3px;'>{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Value metrics
        st.markdown("""
        <div style='background:#0e1018; border:1px solid #1a1d27; border-radius:12px; padding:20px;'>
            <div style='font-size:12px; font-weight:600; color:#565a6e; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:14px;'>Impact on Customer Experience</div>
        """, unsafe_allow_html=True)
        metrics = [("Complaint registered in", "Under 5 seconds"), ("Resolution time improvement", "50–60% faster"), ("Call centre load reduction", "40% drop"), ("Manual effort eliminated", "70%")]
        for label, val in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1a1d27;'>
                <div style='font-size:12px; color:#8890a4;'>{label}</div>
                <div style='font-size:13px; font-weight:600; color:#60a5fa;'>{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='sec-hdr'>Live WhatsApp Simulation</div>", unsafe_allow_html=True)

        # Build WhatsApp chat HTML
        msgs_html = ""
        for m in st.session_state["wa_messages"]:
            cls = "wa-msg-in msg-anim" if m["role"] == "bot" else "wa-msg-out msg-anim"
            text_escaped = m["text"].replace("\n", "<br>")
            msgs_html += f"<div class='{cls}'>{text_escaped}<div class='wa-time'>{m['time']}</div></div>"

        if st.session_state["wa_ticket"]:
            t = st.session_state["wa_ticket"]
            msgs_html += f"""
            <div class='wa-msg-in msg-anim' style='background:#f0f7ff;'>
                <div style='font-size:11px; font-weight:600; color:#075e54; margin-bottom:6px;'>Resolve.AI — Complaint Registered</div>
                <div style='font-size:11px; color:#1a1a1a; line-height:1.7;'>
                    <b>Ticket ID:</b> {t['id']}<br>
                    <b>Category:</b> {t['category']}<br>
                    <b>Severity:</b> <span style='color:#e53935; font-weight:600;'>{t['severity']}</span><br>
                    <b>SLA Target:</b> {t['sla']} hours<br>
                    <b>Status:</b> Agent assigned
                </div>
                <div class='wa-time'>{t['time']}</div>
            </div>"""

        phone_html = f"""
        <div style='display:flex; justify-content:center;'>
            <div class='mobile-frame'>
                <div class='mobile-screen'>
                    <div class='wa-header'>
                        <div class='wa-avatar'>R</div>
                        <div>
                            <div class='wa-name'>Resolve.AI Support</div>
                            <div class='wa-sub'>Union Bank of India — 24x7 Active</div>
                        </div>
                    </div>
                    <div class='wa-body'>{msgs_html}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(phone_html, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Input
        user_input = st.text_input("Type your complaint message", placeholder="e.g. My debit card was blocked without notice…", label_visibility="visible", key="wa_input")
        c1, c2 = st.columns([3,1])
        send = c1.button("Send Message", type="primary", use_container_width=True)
        reset = c2.button("Reset Chat", use_container_width=True)

        if reset:
            for k in ["wa_messages","wa_stage","wa_ticket"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        if send and user_input.strip():
            now = datetime.now().strftime("%H:%M")
            st.session_state["wa_messages"].append({"role": "user", "text": user_input, "time": now})

            with st.spinner("Resolve.AI is analysing your message…"):
                result   = engine.analyze(user_input, df)
                sla      = SLA_MAP[result["severity"]]
                new_id   = f"RES-2026-{random.randint(11000,12999)}"
                response = generate_ai_response(
                    user_input, result["category"], result["severity"],
                    "Valued Customer", "UBOI*****", new_id,
                    hf_key if hf_key else None,
                )

            ai_time = (datetime.now() + timedelta(seconds=2)).strftime("%H:%M")
            st.session_state["wa_messages"].append({"role": "bot", "text": response, "time": ai_time})
            st.session_state["wa_ticket"] = {
                "id": new_id,
                "category": result["category"],
                "severity": result["severity"],
                "sla": sla,
                "time": ai_time,
            }
            st.rerun()


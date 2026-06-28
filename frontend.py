# ============================================================
#  frontend.py  —  NeuralQuant (Huly-inspired)
#  Run:  streamlit run frontend.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

from dashboard import (
    save_user, user_exists, get_all_users,
    load_master, fetch_live,
    calculate_rsi, get_signal, get_price_changes,
    run_prediction, run_validation,
    log_prediction, update_actual_prices, read_prediction_history,
    get_top_gainers, get_top_losers,
    clear_prediction_history,
    # Analysis suite
    get_ath_atl, get_monthly_performance, get_volatility_analysis,
    get_volume_anomalies, get_support_resistance, get_trend_analysis,
    get_price_zones,
)

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralQuant — LSTM Crypto AI",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

# ── session state ─────────────────────────────────────────────
for k, v in [
    ('total_preds', 0), ('correct_preds', 0),
    ('logged_in', False), ('username', ""),
    ('recent_stack', []), ('pred_result', None),
    ('val_result', None), ('dark_mode', True),
    ('lstm_model', None), ('lstm_scaler', None), ('lstm_featured', None),
    ('lstm_feat_scaler', None), ('lstm_close_scaler', None), ('lstm_feat_cols', None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

dark = st.session_state.dark_mode

# ── theme tokens ──────────────────────────────────────────────
if dark:
    BG, BG2, PANEL      = "#0a0e1a", "#0f1424", "#181e2e"
    BORDER               = "rgba(255,255,255,0.07)"
    TEXT, MUTED, TXTW    = "#e8eaf0", "#6b7394", "#ffffff"
    ACCENT, ACCENT2      = "#a78bfa", "#7c6aff"
    UP, DOWN, LIGHTNING  = "#00d4aa", "#ff5e5e", "#c084fc"
else:
    BG, BG2, PANEL      = "#f4f4f8", "#e8e8f0", "#ffffff"
    BORDER               = "rgba(0,0,0,0.08)"
    TEXT, MUTED, TXTW    = "#1a1a2e", "#6b6b8a", "#1a1a2e"
    ACCENT, ACCENT2      = "#7c3aed", "#6d28d9"
    UP, DOWN, LIGHTNING  = "#059669", "#dc2626", "#7c3aed"

# ═════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 0 2rem 0 !important; max-width: 100% !important; }}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: rgba(167,139,250,0.3); border-radius: 3px; }}

@keyframes beamPulse {{ 0%,100%{{opacity:.8;}} 50%{{opacity:1;filter:brightness(1.4);}} }}
@keyframes haloPulse {{ 0%,100%{{opacity:.6;transform:scale(1);}} 50%{{opacity:1;transform:scale(1.06);}} }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.2}} }}
@keyframes boltFlash {{ 0%,80%,100%{{opacity:0;}} 84%{{opacity:1;}} 88%{{opacity:.2;}} 92%{{opacity:.8;}} 96%{{opacity:0;}} }}
@keyframes arcSpin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}

.nq-topbar {{
    position:sticky;top:0;z-index:200;
    display:flex;align-items:center;justify-content:space-between;
    padding:0 32px;height:58px;
    background:{'rgba(10,14,26,0.96)' if dark else 'rgba(244,244,248,0.96)'};
    border-bottom:1px solid {BORDER};backdrop-filter:blur(12px);
}}
.nq-logo-row {{ display:flex;align-items:center;gap:10px; }}
.nq-logo-icon {{
    width:34px;height:34px;border-radius:9px;
    background:linear-gradient(135deg,{ACCENT},#60a5fa);
    display:flex;align-items:center;justify-content:center;font-size:18px;
}}
.nq-logo-txt {{ font-size:20px;font-weight:800;letter-spacing:-0.5px;color:{TXTW}; }}
.nq-logo-txt em {{ color:{ACCENT};font-style:normal; }}
.nq-nav {{ display:flex;gap:8px; }}
.nq-nav a {{
    font-size:13px;font-weight:600;color:{MUTED};cursor:pointer;text-decoration:none;
    letter-spacing:0.3px;padding:6px 16px;border-radius:8px;
    border:1px solid transparent;
    background:transparent;
    transition:all .22s ease;
    position:relative;overflow:hidden;
}}
.nq-nav a::before {{
    content:'';position:absolute;inset:0;border-radius:8px;
    background:radial-gradient(ellipse at center,rgba(167,139,250,.18) 0%,transparent 70%);
    opacity:0;transition:opacity .22s;
}}
.nq-nav a:hover {{
    color:{TXTW};
    border-color:rgba(167,139,250,.30);
    background:rgba(167,139,250,.08);
}}
.nq-nav a:hover::before {{ opacity:1; }}
.nq-nav a.active {{
    color:{ACCENT};
    border-color:rgba(167,139,250,.45);
    background:rgba(167,139,250,.12);
    box-shadow:0 0 14px rgba(167,139,250,.18);
}}
.nq-nav a.active::before {{ opacity:1; }}
.nq-live {{
    display:flex;align-items:center;gap:6px;
    background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.25);
    border-radius:20px;padding:4px 14px;
    font-family:'Space Mono',monospace;font-size:12px;color:{ACCENT};
}}
.live-dot {{ width:7px;height:7px;border-radius:50%;background:{ACCENT};animation:blink 1.4s infinite; }}

.nq-hero {{
    position:relative;min-height:88vh;overflow:hidden;background:{BG};
    display:flex;flex-direction:column;justify-content:center;padding:0 6vw;
}}
.nq-beam {{
    position:absolute;top:-60px;left:50%;transform:translateX(-50%);
    width:2px;height:65vh;
    background:linear-gradient(180deg,transparent 0%,{ACCENT} 35%,#60a5fa 65%,rgba(167,139,250,.3) 85%,transparent 100%);
    filter:blur(1px);z-index:1;animation:beamPulse 3s ease-in-out infinite;
}}
.nq-halo {{
    position:absolute;top:40%;left:50%;transform:translate(-50%,-50%);
    width:560px;height:560px;border-radius:50%;
    background:radial-gradient(ellipse,rgba(124,106,255,.16) 0%,rgba(96,165,250,.08) 35%,transparent 70%);
    z-index:0;animation:haloPulse 4s ease-in-out infinite;
}}
.nq-hero-content {{ position:relative;z-index:2;max-width:680px; }}
.nq-eyebrow {{
    font-family:'Space Mono',monospace;font-size:12px;color:{ACCENT};
    letter-spacing:3px;text-transform:uppercase;margin-bottom:18px;
    display:flex;align-items:center;gap:10px;
}}
.nq-eyebrow::before {{ content:'';width:24px;height:1px;background:{ACCENT}; }}
.nq-hero-h1 {{
    font-size:clamp(56px,7vw,96px);font-weight:800;
    line-height:1.02;letter-spacing:-3px;color:{TXTW};margin-bottom:20px;
}}
.nq-hero-h1 em {{
    font-style:normal;
    background:linear-gradient(135deg,{ACCENT} 0%,#60a5fa 50%,{ACCENT2} 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
}}
.nq-hero-sub {{ font-size:17px;color:{MUTED};line-height:1.7;margin-bottom:36px;max-width:500px; }}
.nq-pill-cta {{
    display:inline-flex;align-items:center;gap:8px;
    background:{'rgba(255,255,255,0.07)' if dark else 'rgba(0,0,0,0.06)'};
    border:1px solid {'rgba(255,255,255,0.18)' if dark else 'rgba(0,0,0,0.15)'};
    color:{TXTW};padding:12px 28px;border-radius:30px;
    font-size:14px;font-weight:700;letter-spacing:.5px;
    cursor:pointer;transition:all .2s;backdrop-filter:blur(6px);
}}
.nq-hero-frame {{
    position:absolute;right:4vw;bottom:0;
    width:50%;max-width:680px;z-index:2;
    background:{'rgba(15,20,36,0.88)' if dark else 'rgba(240,240,248,0.92)'};
    border:1px solid rgba(167,139,250,0.2);border-bottom:none;border-radius:16px 16px 0 0;
    padding:22px;backdrop-filter:blur(14px);box-shadow:0 0 80px rgba(124,106,255,0.14);
}}
.nq-frame-dots {{ display:flex;align-items:center;gap:6px;margin-bottom:16px; }}
.nq-dot {{ width:11px;height:11px;border-radius:50%; }}
.nq-chart-ph {{
    height:200px;
    background:linear-gradient(180deg,rgba(124,106,255,.06) 0%,transparent 100%);
    border-radius:8px;border:1px dashed rgba(167,139,250,.18);
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:8px;color:{MUTED};font-family:'Space Mono',monospace;font-size:13px;
}}
.nq-sec {{
    font-size:12px;font-weight:700;color:{MUTED};
    letter-spacing:2.5px;text-transform:uppercase;
    font-family:'Space Mono',monospace;margin:36px 32px 18px;
    display:flex;align-items:center;gap:14px;
}}
.nq-sec::after {{ content:'';flex:1;height:1px;background:{BORDER}; }}

.lcard {{
    position:relative;overflow:hidden;
    background:{PANEL};border:1px solid {BORDER};
    border-radius:14px;padding:22px 24px;
    transition:all 0.3s ease;
}}
.lcard:hover {{
    transform:translateY(-5px);
    box-shadow:0 10px 30px rgba(124,106,255,0.3),0 0 0 1px {LIGHTNING};
    filter:brightness(1.1);
}}
.lcard.featured {{
    border-color:{LIGHTNING};
    box-shadow:0 0 0 1px {LIGHTNING},0 0 32px rgba(192,132,252,.22),0 0 64px rgba(192,132,252,.09);
    background:linear-gradient(135deg,rgba(124,106,255,.07) 0%,{PANEL} 55%);
}}
.lcard-bolts {{ position:absolute;inset:0;pointer-events:none;border-radius:14px;overflow:hidden; }}
.lcard-bolts svg {{ position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;animation:boltFlash 4s ease-in-out infinite; }}
.lcard-bolts svg:nth-child(2) {{ animation-delay:2s; }}
.lcard-title {{ font-size:16px;font-weight:700;color:{TXTW};margin-bottom:16px;position:relative;z-index:1; }}
.lcard-row {{
    display:flex;justify-content:space-between;align-items:center;
    padding:11px 0;border-bottom:1px solid {BORDER};position:relative;z-index:1;
}}
.lcard-row:last-child {{ border-bottom:none; }}
.lcard-sym {{ font-family:'Space Mono',monospace;font-size:14px;font-weight:700;color:{TXTW};min-width:70px; }}
.lcard-name {{ font-family:'Space Mono',monospace;font-size:13px;color:{MUTED};flex:1;padding:0 12px; }}
.lcard-chg {{ font-family:'Space Mono',monospace;font-size:14px;font-weight:700; }}

.price-banner {{
    position:relative;overflow:hidden;background:{PANEL};
    border:1px solid {BORDER};border-radius:14px;padding:20px 28px;margin-bottom:20px;
    transition:all 0.3s ease;
}}
.price-banner:hover {{ transform:translateY(-2px);box-shadow:0 6px 20px rgba(124,106,255,0.15);border-color:{ACCENT}; }}
.price-banner-glow {{
    position:absolute;inset:0;border-radius:14px;pointer-events:none;
    background:linear-gradient(135deg,rgba(192,132,252,.05) 0%,transparent 50%,rgba(96,165,250,.04) 100%);
    animation:haloPulse 5s ease-in-out infinite;
}}
.price-banner-bolts {{ position:absolute;inset:0;pointer-events:none;overflow:hidden;border-radius:14px; }}
.price-banner-bolts svg {{ position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;animation:boltFlash 6s ease-in-out infinite; }}
.price-banner-label {{ font-family:'Space Mono',monospace;font-size:13px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;position:relative;z-index:1; }}

[data-testid="metric-container"] {{
    background:{PANEL} !important;border:1px solid {BORDER} !important;
    border-radius:12px !important;padding:20px 22px !important;text-align:center !important;
}}
[data-testid="metric-container"] label {{
    font-family:'Space Mono',monospace !important;font-size:11px !important;
    letter-spacing:1.5px !important;text-transform:uppercase !important;color:{MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family:'Space Mono',monospace !important;font-size:28px !important;
    font-weight:700 !important;color:{TXTW} !important;
}}
[data-testid="stMetricDelta"] {{ font-family:'Space Mono',monospace !important;font-size:13px !important; }}

.nq-card {{
    background:{PANEL};border:1px solid {BORDER};border-radius:14px;
    padding:22px 24px;margin-bottom:18px;position:relative;overflow:hidden;
    transition:all 0.3s ease;
}}
.nq-card:hover {{ transform:translateY(-3px);box-shadow:0 8px 25px rgba(124,106,255,0.2);border-color:{ACCENT}; }}
.nq-card::before {{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,{ACCENT},transparent);opacity:.4;
}}
.nq-card-title {{ font-size:16px;font-weight:700;color:{TXTW};margin-bottom:5px; }}
.nq-card-sub {{ font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};margin-bottom:18px; }}

.nq-signal {{
    display:inline-flex;align-items:center;gap:8px;
    font-family:'Space Mono',monospace;font-size:15px;font-weight:700;
    padding:12px 22px;border-radius:10px;letter-spacing:1px;margin:10px 0;
}}
.sig-buy  {{ background:rgba(0,212,170,.10);color:{UP};border:1px solid rgba(0,212,170,.28); }}
.sig-sell {{ background:rgba(255,94,94,.10);color:{DOWN};border:1px solid rgba(255,94,94,.28); }}
.sig-hold {{ background:rgba(251,182,80,.10);color:#fbb650;border:1px solid rgba(251,182,80,.28); }}
.nq-rsi-lbl {{ font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:2px;margin-top:14px; }}
.nq-rsi-val {{ font-family:'Space Mono',monospace;font-size:34px;font-weight:700;color:{TXTW}; }}

.acc-row {{
    display:flex;justify-content:space-between;
    font-family:'Space Mono',monospace;font-size:13px;
    padding:9px 0;border-bottom:1px solid {BORDER};
}}
.acc-row:last-child {{ border-bottom:none; }}
.acc-key {{ color:{MUTED}; }}
.acc-val {{ color:{TXTW};font-weight:700; }}

.pred-box {{
    background:linear-gradient(135deg,rgba(167,139,250,.08),rgba(96,165,250,.05));
    border:1px solid rgba(167,139,250,.28);border-radius:12px;padding:22px 26px;margin-top:14px;
}}
.pred-label {{ font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:2px;text-transform:uppercase; }}
.pred-price {{ font-family:'Space Mono',monospace;font-size:38px;font-weight:700;color:{ACCENT};margin:8px 0; }}
.pred-delta {{ font-family:'Space Mono',monospace;font-size:14px; }}

.bt-box {{
    background:rgba(124,106,255,.06);border:1px solid rgba(124,106,255,.22);
    border-radius:12px;padding:18px 22px;margin-top:14px;
}}
.bt-mape-lbl {{ font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:2px; }}
.bt-mape-val {{ font-family:'Space Mono',monospace;font-size:34px;font-weight:700;color:{ACCENT2}; }}

/* ── analysis cards ── */
.az-card {{
    background:{PANEL};border:1px solid {BORDER};border-radius:14px;
    padding:20px 22px;margin-bottom:16px;position:relative;overflow:hidden;
}}
.az-card::before {{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,{ACCENT2},transparent);opacity:.35;
}}
.az-title {{ font-size:15px;font-weight:700;color:{TXTW};margin-bottom:14px; }}
.az-row {{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 0;border-bottom:1px solid {BORDER};
    font-family:'Space Mono',monospace;font-size:13px;
}}
.az-row:last-child {{ border-bottom:none; }}
.az-key {{ color:{MUTED}; }}
.az-val {{ color:{TXTW};font-weight:700; }}
.az-badge {{
    display:inline-block;padding:4px 12px;border-radius:20px;
    font-family:'Space Mono',monospace;font-size:12px;font-weight:700;
    margin:4px 2px;
}}
.badge-ath  {{ background:rgba(0,212,170,.12);color:{UP};border:1px solid rgba(0,212,170,.3); }}
.badge-atl  {{ background:rgba(255,94,94,.12);color:{DOWN};border:1px solid rgba(255,94,94,.3); }}
.badge-norm {{ background:rgba(167,139,250,.10);color:{ACCENT};border:1px solid rgba(167,139,250,.25); }}
.zone-row {{
    display:flex;align-items:center;gap:10px;padding:9px 0;
    border-bottom:1px solid {BORDER};
    font-family:'Space Mono',monospace;font-size:13px;
}}
.zone-dot {{ width:10px;height:10px;border-radius:50%;flex-shrink:0; }}
.sr-level {{
    display:inline-block;padding:5px 14px;border-radius:8px;margin:3px;
    font-family:'Space Mono',monospace;font-size:13px;font-weight:700;
}}
.sr-res {{ background:rgba(255,94,94,.10);color:{DOWN};border:1px solid rgba(255,94,94,.25); }}
.sr-sup {{ background:rgba(0,212,170,.10);color:{UP};border:1px solid rgba(0,212,170,.25); }}

/* monthly heatmap cell */
.heat-cell {{
    display:inline-block;padding:5px 10px;border-radius:6px;margin:2px;
    font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
    min-width:70px;text-align:center;
}}

.nq-history {{
    background:{BG2};border:1px solid {BORDER};border-radius:10px;padding:18px;
    font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};
    line-height:2;max-height:340px;overflow-y:auto;white-space:pre-wrap;margin:0 32px;
}}
.stButton > button {{
    background:linear-gradient(135deg,{ACCENT},{ACCENT2}) !important;
    color:#fff !important;font-weight:800 !important;
    font-family:'Syne',sans-serif !important;border:none !important;
    border-radius:9px !important;padding:11px 26px !important;
    font-size:14px !important;letter-spacing:.4px !important;
    transition:opacity .2s,transform .15s !important;
}}
.stButton > button:hover {{ opacity:.85 !important;transform:translateY(-1px) !important; }}
[data-testid="stSidebar"] {{ background:{BG2} !important;border-right:1px solid {BORDER} !important; }}
[data-testid="stSidebar"] label {{
    font-family:'Space Mono',monospace !important;font-size:11px !important;
    letter-spacing:1.5px !important;text-transform:uppercase !important;color:{MUTED} !important;
}}
.stTextInput input, div[data-baseweb="select"] {{
    background:{BG2} !important;border:1px solid {BORDER} !important;
    border-radius:8px !important;color:{TEXT} !important;
    font-family:'Space Mono',monospace !important;font-size:13px !important;
}}
.stTabs [data-baseweb="tab-list"] {{ background:transparent !important;border-bottom:1px solid {BORDER} !important; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important;color:{MUTED} !important;
    font-family:'Syne',sans-serif !important;font-size:14px !important;
    font-weight:600 !important;border:none !important;padding:10px 22px !important;
}}
.stTabs [aria-selected="true"] {{ color:{ACCENT} !important;border-bottom:2px solid {ACCENT} !important; }}
.stProgress > div > div {{ background:linear-gradient(90deg,{ACCENT},#60a5fa) !important;border-radius:4px !important; }}
.stProgress > div {{ background:rgba(255,255,255,.06) !important;border-radius:4px !important; }}
.stSuccess {{ background:rgba(0,212,170,.07) !important;border-left:3px solid {UP} !important;border-radius:8px !important; }}
.stError   {{ background:rgba(255,94,94,.07) !important;border-left:3px solid {DOWN} !important;border-radius:8px !important; }}
.stInfo    {{ background:rgba(167,139,250,.07) !important;border-left:3px solid {ACCENT} !important;border-radius:8px !important; }}
.stWarning {{ background:rgba(251,182,80,.07) !important;border-left:3px solid #fbb650 !important;border-radius:8px !important; }}
hr {{ border-color:{BORDER} !important;margin:20px 0 !important; }}
[data-testid="stDataFrame"] {{ border-radius:10px !important;overflow:hidden !important; }}

.nq-clock-section {{
    position:relative;overflow:hidden;background:{BG2};border-top:1px solid {BORDER};
    min-height:320px;display:flex;align-items:center;justify-content:center;
    gap:80px;padding:60px 40px;
}}
.nq-clock-section::before {{
    content:'';position:absolute;top:-30%;left:-8%;width:52%;height:160%;
    background:linear-gradient(125deg,rgba(234,88,12,.16) 0%,rgba(124,106,255,.1) 45%,transparent 70%);
    transform:skewX(-14deg);pointer-events:none;
}}
.nq-clock-wrap {{ position:relative;width:220px;height:220px;flex-shrink:0; }}
.nq-clock-face {{
    width:220px;height:220px;border-radius:50%;
    background:radial-gradient(circle,#1c1c2e 60%,#0f0f1a 100%);
    border:2px solid rgba(255,255,255,.08);
    box-shadow:0 0 0 8px rgba(0,0,0,.4),0 0 40px rgba(124,106,255,.2);
    position:relative;overflow:hidden;
}}
.nq-arc-orange {{
    position:absolute;inset:-3px;border-radius:50%;
    background:conic-gradient(from 0deg,transparent 0%,rgba(234,88,12,.85) 25%,transparent 30%);
    animation:arcSpin 8s linear infinite;
}}
.nq-arc-blue {{
    position:absolute;inset:-3px;border-radius:50%;
    background:conic-gradient(from 180deg,transparent 0%,rgba(96,165,250,.85) 20%,{ACCENT} 26%,transparent 31%);
    animation:arcSpin 8s linear infinite;
}}
.nq-clock-inner {{
    position:absolute;inset:7px;border-radius:50%;background:#12121e;
    display:flex;align-items:center;justify-content:center;
}}
.nq-clock-center {{ position:relative;width:100%;height:100%; }}
.nq-hand {{ position:absolute;bottom:50%;left:50%;transform-origin:bottom center;border-radius:4px; }}
.nq-hand-h {{ width:4px;height:52px;margin-left:-2px;background:rgba(255,255,255,.9); }}
.nq-hand-m {{ width:3px;height:70px;margin-left:-1.5px;background:rgba(255,255,255,.7); }}
.nq-hand-s {{ width:2px;height:76px;margin-left:-1px;background:{ACCENT}; }}
.nq-clock-dot {{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    width:10px;height:10px;border-radius:50%;
    background:{ACCENT};box-shadow:0 0 10px {ACCENT};z-index:10;
}}
.nq-cta-block {{ max-width:420px; }}
.nq-cta-eye {{ font-family:'Space Mono',monospace;font-size:11px;color:{ACCENT};letter-spacing:3px;text-transform:uppercase;margin-bottom:14px; }}
.nq-cta-h2 {{ font-size:clamp(30px,3.8vw,50px);font-weight:800;line-height:1.06;letter-spacing:-1.5px;color:{TXTW};margin-bottom:14px; }}
.nq-cta-h2 em {{ color:{ACCENT};font-style:normal; }}
.nq-cta-sub {{ font-size:15px;color:{MUTED};line-height:1.7;margin-bottom:24px; }}
.nq-cta-btns {{ display:flex;gap:12px;flex-wrap:wrap; }}
.btn-outline {{
    display:inline-flex;align-items:center;gap:6px;
    background:{'rgba(255,255,255,.06)' if dark else 'rgba(0,0,0,.05)'};
    border:1px solid {'rgba(255,255,255,.2)' if dark else 'rgba(0,0,0,.15)'};
    color:{TXTW};padding:11px 26px;border-radius:30px;
    font-size:14px;font-weight:600;cursor:pointer;font-family:'Syne',sans-serif;transition:all .2s;
}}
.btn-outline:hover {{ border-color:{ACCENT};color:{ACCENT}; }}
.btn-solid {{
    display:inline-flex;align-items:center;gap:6px;
    background:{ACCENT};border:1px solid {ACCENT};color:#fff;padding:11px 26px;border-radius:30px;
    font-size:14px;font-weight:700;cursor:pointer;font-family:'Syne',sans-serif;transition:all .2s;
}}
.btn-solid:hover {{ opacity:.85; }}
.nq-footer {{
    background:{BG};border-top:1px solid {BORDER};
    padding:18px 32px;display:flex;justify-content:space-between;align-items:center;
}}
.nq-footer-l {{ font-family:'Space Mono',monospace;font-size:12px;color:{MUTED}; }}
.nq-footer-r {{ display:flex;gap:22px; }}
.nq-footer-r a {{ font-size:12px;color:{MUTED};text-decoration:none;transition:color .2s; }}
.nq-footer-r a:hover {{ color:{ACCENT}; }}
.cp {{ padding:0 32px; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ═════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("<style>[data-testid='stSidebar']{display:none!important}</style>",
                unsafe_allow_html=True)
    st.markdown(f"""
<div style="position:fixed;inset:0;background:{BG};z-index:-1;"></div>
<div style="position:fixed;top:-80px;left:50%;transform:translateX(-50%);
    width:2px;height:60vh;
    background:linear-gradient(180deg,transparent 0%,{ACCENT} 40%,#60a5fa 70%,transparent 100%);
    filter:blur(1px);animation:beamPulse 3s ease-in-out infinite;z-index:0;"></div>
<div style="position:fixed;top:35%;left:50%;transform:translate(-50%,-50%);
    width:480px;height:480px;border-radius:50%;
    background:radial-gradient(ellipse,rgba(124,106,255,.15) 0%,transparent 70%);
    z-index:0;animation:haloPulse 4s ease-in-out infinite;"></div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;
    padding:80px 0 28px;position:relative;z-index:1;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
        <div style="width:42px;height:42px;border-radius:11px;
            background:linear-gradient(135deg,{ACCENT},#60a5fa);
            display:flex;align-items:center;justify-content:center;font-size:22px;">⚡</div>
        <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px;color:{TXTW};">
            Neural<em style="color:{ACCENT};font-style:normal;">Quant</em></div>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};
        letter-spacing:.5px;margin-bottom:32px;">
        LSTM-powered crypto intelligence
    </div>
</div>
""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"""
<div style="background:{'rgba(24,30,46,0.88)' if dark else 'rgba(255,255,255,0.92)'};
    border:1px solid rgba(167,139,250,.2);border-radius:20px;
    padding:36px 40px;backdrop-filter:blur(16px);
    box-shadow:0 8px 60px rgba(0,0,0,.4);position:relative;z-index:1;">
""", unsafe_allow_html=True)

        login_tab, reg_tab = st.tabs(["🔑  Sign In", "✨  Register"])
        with login_tab:
            lu = st.text_input("Username", key="lu", placeholder="your username")
            lp = st.text_input("Password", type="password", key="lp", placeholder="••••••••")
            if st.button("Sign In →", key="btn_login"):
                if user_exists(lu, lp):
                    st.session_state.logged_in = True
                    st.session_state.username  = lu
                    update_actual_prices(lu)
                    st.success(f"Welcome back, {lu}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with reg_tab:
            ru = st.text_input("Choose Username", key="ru", placeholder="pick a username")
            rp = st.text_input("Choose Password", type="password", key="rp", placeholder="••••••••")
            if st.button("Create Account →", key="btn_reg"):
                if not ru.strip() or not rp.strip():
                    st.warning("Please fill all fields.")
                elif ru in get_all_users():
                    st.error("Username already taken.")
                else:
                    save_user(ru, rp)
                    st.success("Account created! Sign in above.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════
#  FORCE SIDEBAR VISIBLE after login
#  The login page injects display:none on the sidebar.
#  That CSS persists in the DOM across reruns, so we
#  must explicitly re-show it here for logged-in users.
# ═════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarNav"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)
@st.cache_data
def cached_master():
    return load_master()

@st.cache_data
def cached_live(sym):
    return fetch_live(sym)

master_df = cached_master()


# ═════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    search        = st.text_input("Search Coin", "")
    filtered      = master_df[master_df['Coin Name'].str.contains(search, case=False, na=False)]
    selected_name = st.selectbox("Select Asset", filtered['Coin Name'].tolist())
    sel_info      = master_df[master_df['Coin Name'] == selected_name].iloc[0]
    symbol        = str(sel_info['Symbol']).strip()

    st.markdown("---")
    st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:10px;
    color:{MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
    Coin Directory</div>
<div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};line-height:2.3;">
    <div style="color:{TXTW};font-weight:700;margin-bottom:6px;">{selected_name}</div>
    Symbol: <span style="color:{ACCENT};">{symbol}</span><br>
    Stack depth: <span style="color:{ACCENT2};">{len(st.session_state.recent_stack)}</span><br>
    Total preds: <span style="color:#fbb650;">{st.session_state.total_preds}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
<div style="padding:16px 0 8px;">
    <div style="font-family:'Space Mono',monospace;font-size:10px;
        color:{MUTED};letter-spacing:2px;text-transform:uppercase;">Logged in as</div>
    <div style="font-size:17px;font-weight:700;color:{ACCENT};margin-top:4px;">
        {st.session_state.username}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    mode_lbl = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(mode_lbl, key="toggle_mode"):
        st.session_state.dark_mode = not dark
        st.rerun()
    if st.button("⇠ Logout", key="btn_logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()


# ═════════════════════════════════════════════════════════════
#  TOPBAR
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="nq-topbar">
    <div class="nq-logo-row">
        <div class="nq-logo-icon">⚡</div>
        <div class="nq-logo-txt">Neural<em>Quant</em></div>
    </div>
    <div class="nq-nav">
        <a class="active" href="#section-dashboard" onclick="document.getElementById('section-dashboard').scrollIntoView({{behavior:'smooth'}});return false;">Dashboard</a>
        <a href="#section-lstm" onclick="document.getElementById('section-lstm').scrollIntoView({{behavior:'smooth'}});return false;">Predictions</a>
        <a href="#section-analysis" onclick="document.getElementById('section-analysis').scrollIntoView({{behavior:'smooth'}});return false;">Analysis</a>
        <a href="#section-movers" onclick="document.getElementById('section-movers').scrollIntoView({{behavior:'smooth'}});return false;">Movers</a>
        <a href="#section-history" onclick="document.getElementById('section-history').scrollIntoView({{behavior:'smooth'}});return false;">History</a>
    </div>
    <div class="nq-live"><span class="live-dot"></span>&nbsp;MARKET LIVE</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="nq-hero">
    <div class="nq-beam"></div>
    <div class="nq-halo"></div>
    <div class="nq-hero-content">
        <div class="nq-eyebrow">LSTM Neural Intelligence</div>
        <h1 class="nq-hero-h1">Neural<em>Quant</em><br>Predict Before<br>the Market Moves</h1>
        <p class="nq-hero-sub">
            Advanced LSTM deep learning on live crypto data —
            RSI signals, price forecasts and backtested accuracy in real time.
        </p>
        <span class="nq-pill-cta">SEE IN ACTION &nbsp;→</span>
    </div>
    <div class="nq-hero-frame">
        <div class="nq-frame-dots">
            <div class="nq-dot" style="background:#ff5f57;"></div>
            <div class="nq-dot" style="background:#febc2e;"></div>
            <div class="nq-dot" style="background:#28c840;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:13px;
                color:{MUTED};margin-left:8px;">{selected_name} · Live Chart</span>
        </div>
        <div class="nq-chart-ph">
            <span style="font-size:26px;">📈</span>
            <span>// live chart renders below</span>
            <span style="opacity:.5;font-size:11px;">OHLCV · RSI · MACD</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  FETCH LIVE DATA  (needed for hero sparkline)
# ═════════════════════════════════════════════════════════════
hist_df = cached_live(symbol)

# ─── build a tiny SVG sparkline from last 90 days of Close ───
def make_sparkline_svg(df, accent, muted, bg2, border):
    """Returns an inline SVG sparkline string (700×180px viewBox)."""
    if df.empty or len(df) < 5:
        return ""
    close = df['Close'].squeeze().dropna().tail(90).values.astype(float)
    n     = len(close)
    lo, hi = close.min(), close.max()
    rng   = hi - lo if hi != lo else 1.0

    W, H, PAD = 660, 140, 10
    xs = [PAD + (i / (n - 1)) * (W - PAD * 2) for i in range(n)]
    ys = [PAD + (1 - (v - lo) / rng) * (H - PAD * 2) for v in close]

    # polyline path
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # filled area path
    area = f"M{xs[0]:.1f},{ys[0]:.1f} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs[1:], ys[1:]))
    area += f" L{xs[-1]:.1f},{H+PAD} L{xs[0]:.1f},{H+PAD} Z"

    # last price dot
    lx, ly = xs[-1], ys[-1]
    direction_up = close[-1] >= close[0]
    line_color   = accent

    return f"""
<svg viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;display:block;">
  <defs>
    <linearGradient id="spkGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{accent}" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <!-- grid lines -->
  <line x1="10" y1="50" x2="670" y2="50" stroke="{border}" stroke-width="1"/>
  <line x1="10" y1="90" x2="670" y2="90" stroke="{border}" stroke-width="1"/>
  <line x1="10" y1="130" x2="670" y2="130" stroke="{border}" stroke-width="1"/>
  <!-- area fill -->
  <path d="{area}" fill="url(#spkGrad)" transform="translate(10,0)"/>
  <!-- line -->
  <polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="2.2"
    stroke-linejoin="round" stroke-linecap="round" transform="translate(10,0)"/>
  <!-- last dot pulse -->
  <circle cx="{lx+10:.1f}" cy="{ly:.1f}" r="6" fill="{accent}" opacity="0.25"/>
  <circle cx="{lx+10:.1f}" cy="{ly:.1f}" r="3.5" fill="{accent}"/>
  <!-- price labels -->
  <text x="670" y="16" text-anchor="end"
    font-family="Space Mono,monospace" font-size="11" fill="{muted}">HIGH ${hi:,.0f}</text>
  <text x="670" y="155" text-anchor="end"
    font-family="Space Mono,monospace" font-size="11" fill="{muted}">LOW ${lo:,.0f}</text>
</svg>"""

sparkline_svg = make_sparkline_svg(hist_df, ACCENT, MUTED, BG2, BORDER) if not hist_df.empty else ""
sparkline_html = sparkline_svg if sparkline_svg else f"""
<div style="height:160px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:8px;color:{MUTED};
    font-family:'Space Mono',monospace;font-size:12px;">
    <span style="font-size:22px;">📡</span>
    <span>No data — check symbol</span>
</div>"""

# ═════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div id="section-top" class="nq-hero">
    <div class="nq-beam"></div>
    <div class="nq-halo"></div>
    <div class="nq-hero-content">
        <div class="nq-eyebrow">LSTM Neural Intelligence</div>
        <h1 class="nq-hero-h1">Neural<em>Quant</em><br>Predict Before<br>the Market Moves</h1>
        <p class="nq-hero-sub">
            Advanced LSTM deep learning on live crypto data —
            RSI signals, price forecasts and backtested accuracy in real time.
        </p>
        <a class="nq-pill-cta" href="#section-dashboard"
            onclick="document.getElementById('section-dashboard').scrollIntoView({{behavior:'smooth'}});return false;">
            SEE IN ACTION &nbsp;→
        </a>
    </div>
    <div class="nq-hero-frame">
        <div class="nq-frame-dots">
            <div class="nq-dot" style="background:#ff5f57;"></div>
            <div class="nq-dot" style="background:#febc2e;"></div>
            <div class="nq-dot" style="background:#28c840;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:13px;
                color:{MUTED};margin-left:8px;">{selected_name} · 90-Day Close</span>
        </div>
        <div style="height:160px;overflow:hidden;border-radius:8px;
            border:1px solid rgba(167,139,250,.15);background:rgba(167,139,250,.03);">
            {sparkline_html}
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:10px;
            font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">
            <span>90 days ago</span>
            <span style="color:{ACCENT};">▲ Live · {symbol}-USD</span>
            <span>Today</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div id="section-movers" class="nq-sec">Top Market Movers · 24h</div>', unsafe_allow_html=True)

gainers = get_top_gainers(master_df)
losers  = get_top_losers(master_df)

def mover_rows(df, color):
    html = ""
    for _, row in df.iterrows():
        sym  = str(row['Symbol'])[:10]
        name = str(row['Coin Name'])[:22]
        chg  = float(row['24h'])
        sign = "+" if chg > 0 else ""
        html += f"""
<div class="lcard-row">
    <span class="lcard-sym">{sym}</span>
    <span class="lcard-name">{name}</span>
    <span class="lcard-chg" style="color:{color};">{sign}{chg:.2f}%</span>
</div>"""
    return html

g_rows = mover_rows(gainers, UP)
l_rows = mover_rows(losers,  DOWN)
col_g, col_l = st.columns(2)

with col_g:
    st.markdown(f"""
<div style="padding:0 0 0 32px;">
<div class="lcard featured">
    <div class="lcard-bolts">
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="2,0 2,80 9,115 2,145 2,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
            <polyline points="398,0 398,65 391,105 398,135 391,185 398,235 398,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
        </svg>
    </div>
    <div class="lcard-title">🟢 Top Gainers</div>
    {g_rows}
</div>
</div>
""", unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
<div style="padding:0 32px 0 0;">
<div class="lcard">
    <div class="lcard-bolts">
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="2,0 2,80 9,115 2,145 2,300" stroke="{LIGHTNING}" stroke-width="2" fill="none"/>
        </svg>
    </div>
    <div class="lcard-title">🔴 Top Losers</div>
    {l_rows}
</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  MAIN HEADING
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center;padding:60px 6vw 40px;background:{BG};">
    <div class="nq-eyebrow" style="justify-content:center;">LSTM Neural Intelligence</div>
    <h1 style="font-size:clamp(60px,8vw,110px);font-weight:800;line-height:1.02;
        letter-spacing:-3px;color:{TXTW};margin-bottom:20px;">
        Neural<em style="background:linear-gradient(135deg,{ACCENT} 0%,#60a5fa 50%,{ACCENT2} 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;font-style:normal;">Quant</em><br>
        <span style="font-size:0.82em;">Predict Before<br>the Market Moves</span>
    </h1>
    <p style="font-size:20px;color:{MUTED};line-height:1.7;max-width:640px;margin:0 auto;">
        Advanced LSTM deep learning on live crypto data —
        RSI signals, price forecasts and backtested accuracy in real time.
    </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  MAIN TABS
# ═════════════════════════════════════════════════════════════
tab_dash, tab_analysis = st.tabs(["📊  Dashboard", "🔬  Analysis"])


# ─────────────────────────────────────────────────────────────
#  TAB 1 — DASHBOARD  (original content)
# ─────────────────────────────────────────────────────────────
with tab_dash:

    if not hist_df.empty:
        prices = get_price_changes(hist_df)

        st.markdown('<div id="section-dashboard" class="nq-sec">Price Change Indicators</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="cp">
<div class="price-banner">
    <div class="price-banner-glow"></div>
    <div class="price-banner-bolts">
        <svg viewBox="0 0 900 80" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <polyline points="0,0 0,25 7,42 0,58 0,80" stroke="{LIGHTNING}" stroke-width="1.8" fill="none"/>
            <polyline points="900,0 900,30 893,50 900,65 900,80" stroke="{LIGHTNING}" stroke-width="1.8" fill="none"/>
        </svg>
    </div>
    <div class="price-banner-label">⚡ Live Market Data · {selected_name} ({symbol})</div>
</div>
</div>
""", unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Price", f"${prices['current']:,.2f}")
        col_m2.metric("24h Change",    f"{prices['change_24h']:.2f}%", delta=f"{prices['change_24h']:.2f}%")
        col_m3.metric("7d Change",     f"{prices['change_7d']:.2f}%",  delta=f"{prices['change_7d']:.2f}%")

        st.markdown('<div class="nq-sec">Historical Price Chart</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="cp"><div class="nq-card">
    <div class="nq-card-title">{selected_name} · Close Price</div>
    <div class="nq-card-sub">Daily closing prices · Source: Yahoo Finance</div>
</div></div>
""", unsafe_allow_html=True)
        st.line_chart(hist_df['Close'], use_container_width=True)

        # RSI SIGNAL + ACCURACY
        st.markdown('<div class="nq-sec">Technical Signal · Accuracy Tracker</div>',
                    unsafe_allow_html=True)

        rsi_vals   = calculate_rsi(hist_df['Close'])
        latest_rsi = float(rsi_vals.iloc[-1].squeeze())
        signal     = get_signal(latest_rsi)
        sig_cls    = {"BUY":"sig-buy","SELL":"sig-sell","HOLD":"sig-hold"}[signal]
        sig_txt    = {"BUY":"🟢 BUY — Oversold","SELL":"🔴 SELL — Overbought","HOLD":"🟡 HOLD — Neutral"}[signal]

        total   = st.session_state.total_preds
        correct = st.session_state.correct_preds
        acc_pct = (correct / total * 100) if total > 0 else 0

        cs, ca = st.columns(2)
        with cs:
            st.markdown(f"""
<div class="cp"><div class="nq-card">
    <div class="nq-card-title">RSI Signal</div>
    <div class="nq-card-sub">Relative Strength Index · 14-period</div>
    <div class="nq-signal {sig_cls}">{sig_txt}</div>
    <div class="nq-rsi-lbl">CURRENT RSI</div>
    <div class="nq-rsi-val">{latest_rsi:.2f}</div>
</div></div>
""", unsafe_allow_html=True)

        with ca:
            st.markdown(f"""
<div class="cp"><div class="nq-card">
    <div class="nq-card-title">Accuracy Tracker</div>
    <div class="nq-card-sub">Session prediction performance</div>
    <div class="acc-row"><span class="acc-key">Total Predictions</span><span class="acc-val">{total}</span></div>
    <div class="acc-row"><span class="acc-key">Correct (MAPE &lt; 5%)</span><span class="acc-val">{correct}</span></div>
    <div class="acc-row"><span class="acc-key">Session Accuracy</span>
        <span class="acc-val" style="color:{ACCENT};">{acc_pct:.1f}%</span></div>
</div></div>
""", unsafe_allow_html=True)
            st.progress(acc_pct / 100)

        # LSTM ENGINE
        st.markdown('<div id="section-lstm" class="nq-sec">LSTM Engine · 19-Feature BiLSTM</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
<div class="cp"><div class="nq-card">
    <div class="nq-card-title">🤖 Tomorrow's Price Prediction</div>
    <div class="nq-card-sub">Bidirectional LSTM · 90-day lookback · 19 features · Huber loss · RobustScaler</div>
</div></div>
""", unsafe_allow_html=True)

        if st.button("⚡ Predict Tomorrow", key="btn_predict", use_container_width=True):
            with st.spinner("Training LSTM model (17 features, 60-day window)…"):
                st.session_state.total_preds += 1
                result = run_prediction(hist_df, symbol=symbol, master_df=master_df)
                pred   = result["price"]
                if not np.isfinite(pred) or pred <= 0:
                    pred = prices['current']
                    st.warning("⚠ Model returned invalid prediction — showing current price.")
                st.session_state.lstm_model        = result["model"]
                st.session_state.lstm_feat_scaler  = result["feat_scaler"]
                st.session_state.lstm_close_scaler = result["close_scaler"]
                st.session_state.lstm_featured     = result["featured"]
                st.session_state.lstm_feat_cols    = result["feat_cols"]
                st.session_state.recent_stack.append(pred)
                log_prediction(st.session_state.username, symbol, pred)
                st.session_state.pred_result = pred

        if st.session_state.pred_result is not None:
            pred  = st.session_state.pred_result
            diff  = pred - prices['current']
            clr   = UP if diff >= 0 else DOWN
            arrow = "▲" if diff >= 0 else "▼"
            st.markdown(f"""
<div class="cp"><div class="pred-box">
    <div class="pred-label">AI Predicted Close Price</div>
    <div class="pred-price">${pred:,.2f}</div>
    <div class="pred-delta" style="color:{clr};">{arrow} {diff:+,.2f} vs current</div>
</div></div>
""", unsafe_allow_html=True)
            if st.session_state.recent_stack:
                top = st.session_state.recent_stack[-1]
                st.markdown(f"""
<div class="cp"><div style="margin-top:10px;font-family:'Space Mono',monospace;font-size:13px;
    color:{MUTED};padding:11px 16px;background:{BG2};border-radius:8px;border:1px solid {BORDER};">
    Stack top → <span style="color:{ACCENT2};">${top:,.2f}</span>
    &nbsp;·&nbsp; depth: {len(st.session_state.recent_stack)}
</div></div>
""", unsafe_allow_html=True)
            st.success("Prediction logged. Actual price will auto-update tomorrow.")

        # VALIDATION
        st.markdown(f"""
<div class="cp"><div class="nq-card">
    <div class="nq-card-title">📊 Validate Accuracy</div>
    <div class="nq-card-sub">Predicted vs Actual · last 30 days · instant (no retraining)</div>
</div></div>
""", unsafe_allow_html=True)

        if st.button("📊 Validate Accuracy", key="btn_validate", use_container_width=True):
            with st.spinner("Comparing predicted vs actual prices…"):
                from dashboard import validate_prediction_history
                result = validate_prediction_history(st.session_state.username)
                st.session_state.val_result = result

        if st.session_state.val_result is not None:
            res  = st.session_state.val_result
            mape = res["mape"]
            
            # Handle case where mape is None (insufficient validation data)
            if mape is None:
                st.warning(res.get("message", "⚠ Not enough completed predictions yet. Please generate and wait for more predictions to complete."))
            else:
                clr  = UP if mape < 5 else "#fbb650" if mape < 10 else DOWN
                msg  = "✅ High accuracy" if mape < 5 else "⚠ Moderate deviation" if mape < 10 else "⛔ High deviation"
                st.markdown(f"""
<div class="cp"><div class="bt-box">
    <div class="bt-mape-lbl">MAPE · ALL COMPLETED PREDICTIONS</div>
    <div class="bt-mape-val" style="color:{clr};">{mape:.2f}%</div>
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};margin-top:6px;">{msg} · {res.get("count", 0)} predictions</div>
</div></div>
""", unsafe_allow_html=True)
                
                comp_df = res.get("comp_df", pd.DataFrame())
                if not comp_df.empty:
                    st.markdown(f"""
<div class="cp" style="margin-top:14px;">
<div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};
    letter-spacing:1px;margin-bottom:6px;">Predicted vs Actual Price</div>
</div>
""", unsafe_allow_html=True)
                    # Display chart with Predicted and Actual columns
                    st.line_chart(comp_df[["Predicted", "Actual"]], use_container_width=True)
                    
                    st.markdown(f"""
<div class="cp" style="margin-top:6px;">
<div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};
    letter-spacing:1px;margin-bottom:6px;">Absolute Error per Prediction</div>
</div>
""", unsafe_allow_html=True)
                    st.bar_chart(comp_df["Error"], use_container_width=True)
                    
                    # Show detailed table
                    st.markdown(f"""
<div class="cp" style="margin-top:14px;">
<div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};
    letter-spacing:1px;margin-bottom:6px;">Detailed Results</div>
</div>
""", unsafe_allow_html=True)
                    display_df = comp_df.copy()
                    display_df["MAPE"] = display_df["MAPE"].round(2)
                    display_df["Predicted"] = display_df["Predicted"].round(2)
                    display_df["Actual"] = display_df["Actual"].round(2)
                    display_df["Error"] = display_df["Error"].round(2)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.markdown('<div class="cp">', unsafe_allow_html=True)
        st.warning("⚠ No valid market data found for this coin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # CLEAR PREDICTION HISTORY
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🗑 Clear Prediction History", use_container_width=True):
            success = clear_prediction_history(st.session_state.username)
            if success:
                st.success("Prediction history deleted successfully.")
                st.session_state.val_result = None
            else:
                st.warning("No prediction history found.")

    # PREDICTION HISTORY
    st.markdown('<div id="section-history" class="nq-sec">Prediction History</div>', unsafe_allow_html=True)
    history_text = read_prediction_history(st.session_state.username)
    if history_text:
        st.markdown(f'<div class="nq-history">{history_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="nq-history" style="text-align:center;padding:32px;color:{MUTED};">
    No predictions yet. Run your first LSTM prediction above!
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 2 — ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab_analysis:

    if hist_df.empty:
        st.warning("⚠ No market data available for analysis.")
    else:
        st.markdown(f"""
<div id="section-analysis" style="padding:28px 32px 4px;">
<div style="font-size:22px;font-weight:800;color:{TXTW};">
    {selected_name} <span style="color:{ACCENT};font-size:16px;font-family:'Space Mono',monospace;">({symbol})</span>
    — Deep Historical Analysis</div>
<div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};margin-top:6px;">
    Full dataset from 2020-01-01 to today · all zones, trends, and anomalies
</div>
</div>
""", unsafe_allow_html=True)

        # ── ROW 1: Trend + ATH/ATL ────────────────────────────
        st.markdown('<div class="nq-sec">Market Regime · All-Time Levels</div>',
                    unsafe_allow_html=True)

        trend_data = get_trend_analysis(hist_df)
        ath_data   = get_ath_atl(hist_df)
        sr_data    = get_support_resistance(hist_df)

        col_t, col_a, col_sr = st.columns(3)

        with col_t:
            td = trend_data
            tc = UP if "Bull" in td["trend"] else DOWN if "Bear" in td["trend"] else "#fbb650"
            st.markdown(f"""
<div class="cp"><div class="az-card">
    <div class="az-title">📡 Trend Analysis</div>
    <div style="font-size:20px;font-weight:800;color:{tc};margin-bottom:14px;">{td["trend"]}</div>
    <div class="az-row">
        <span class="az-key">EMA-20</span>
        <span class="az-val">${td["ema20"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">EMA-50</span>
        <span class="az-val">${td["ema50"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">EMA-200</span>
        <span class="az-val">${td["ema200"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">Trend Strength</span>
        <span class="az-val" style="color:{ACCENT};">{td["trend_strength"]:.2f}%</span>
    </div>
    <div class="az-row">
        <span class="az-key">Days in Regime</span>
        <span class="az-val">{td["ema_streak_days"]}</span>
    </div>
</div></div>
""", unsafe_allow_html=True)

        with col_a:
            ad = ath_data
            ath_clr = DOWN if ad["pct_from_ath"] < -30 else "#fbb650" if ad["pct_from_ath"] < -10 else UP
            st.markdown(f"""
<div class="cp"><div class="az-card">
    <div class="az-title">🏔 All-Time High / Low</div>
    <div class="az-row">
        <span class="az-key">Current Price</span>
        <span class="az-val">${ad["current"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">All-Time High</span>
        <span class="az-val" style="color:{UP};">${ad["ath"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">ATH Date</span>
        <span class="az-val">{ad["ath_date"]}</span>
    </div>
    <div class="az-row">
        <span class="az-key">% from ATH</span>
        <span class="az-val" style="color:{ath_clr};">{ad["pct_from_ath"]:.1f}%</span>
    </div>
    <div class="az-row">
        <span class="az-key">All-Time Low</span>
        <span class="az-val" style="color:{DOWN};">${ad["atl"]:,.2f}</span>
    </div>
    <div class="az-row">
        <span class="az-key">ATL Date</span>
        <span class="az-val">{ad["atl_date"]}</span>
    </div>
    <div class="az-row">
        <span class="az-key">% from ATL</span>
        <span class="az-val" style="color:{UP};">+{ad["pct_from_atl"]:.1f}%</span>
    </div>
</div></div>
""", unsafe_allow_html=True)

        with col_sr:
            res_levels = sr_data["resistance"]
            sup_levels = sr_data["support"]
            res_html   = "".join([f'<span class="sr-level sr-res">${v:,.2f}</span>' for v in res_levels]) or f'<span style="color:{MUTED};">None found</span>'
            sup_html   = "".join([f'<span class="sr-level sr-sup">${v:,.2f}</span>' for v in sup_levels]) or f'<span style="color:{MUTED};">None found</span>'
            st.markdown(f"""
<div class="cp"><div class="az-card">
    <div class="az-title">🎯 Support & Resistance</div>
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};
        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Resistance Levels</div>
    <div style="margin-bottom:18px;">{res_html}</div>
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};
        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Support Levels</div>
    <div>{sup_html}</div>
    <div class="az-row" style="margin-top:14px;">
        <span class="az-key">Current Price</span>
        <span class="az-val" style="color:{ACCENT};">${sr_data["current"]:,.2f}</span>
    </div>
</div></div>
""", unsafe_allow_html=True)

        # ── ROW 2: Price Zones Chart ───────────────────────────
        st.markdown('<div class="nq-sec">Price Zone History (ATH / High / Normal / Low / ATL)</div>',
                    unsafe_allow_html=True)

        zones_df = get_price_zones(hist_df)

        zone_colors = {
            "ATH Zone": UP,
            "High":     "#60a5fa",
            "Normal":   ACCENT,
            "Low":      "#fbb650",
            "ATL Zone": DOWN,
        }

        st.markdown(f"""
<div class="cp"><div class="az-card">
    <div class="az-title">📊 Historical Price Zones</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
""", unsafe_allow_html=True)

        for zone_name, zc in zone_colors.items():
            st.markdown(f"""
<span style="display:inline-flex;align-items:center;gap:6px;
    font-family:'Space Mono',monospace;font-size:12px;color:{zc};">
    <span style="width:10px;height:10px;border-radius:50%;background:{zc};display:inline-block;"></span>
    {zone_name}
</span>
""", unsafe_allow_html=True)

        st.markdown("</div></div></div>", unsafe_allow_html=True)

        st.line_chart(zones_df.set_index('Date')['Close'], use_container_width=True)

        # Zone count summary
        zone_counts = zones_df['Zone'].value_counts()
        total_days  = len(zones_df)
        zone_row_html = ""
        for zone_name, zc in zone_colors.items():
            cnt = zone_counts.get(zone_name, 0)
            pct = cnt / total_days * 100
            zone_row_html += f"""
<div class="zone-row">
    <span class="zone-dot" style="background:{zc};"></span>
    <span style="color:{TXTW};font-weight:700;min-width:100px;">{zone_name}</span>
    <span style="color:{MUTED};flex:1;">{cnt} days</span>
    <span style="color:{zc};font-weight:700;">{pct:.1f}%</span>
</div>"""

        st.markdown(f"""
<div class="cp"><div class="az-card">
    <div class="az-title">📋 Zone Distribution Summary</div>
    {zone_row_html}
</div></div>
""", unsafe_allow_html=True)

        # ── ROW 3: Monthly Performance Heatmap ────────────────
        st.markdown('<div class="nq-sec">Monthly Performance Heatmap</div>',
                    unsafe_allow_html=True)

        monthly_df = get_monthly_performance(hist_df)

        def heat_color(r):
            if r >= 20:  return f"background:rgba(0,212,170,.25);color:{UP};border:1px solid {UP};"
            if r >= 5:   return f"background:rgba(96,165,250,.15);color:#60a5fa;border:1px solid #60a5fa;"
            if r >= 0:   return f"background:rgba(167,139,250,.10);color:{ACCENT};border:1px solid {ACCENT};"
            if r >= -5:  return f"background:rgba(251,182,80,.10);color:#fbb650;border:1px solid #fbb650;"
            if r >= -20: return f"background:rgba(255,94,94,.12);color:{DOWN};border:1px solid {DOWN};"
            return             f"background:rgba(255,0,0,.18);color:#ff2020;border:1px solid #ff2020;"

        MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        years  = sorted(monthly_df['Year'].unique())

        heat_html = f"""
<div class="cp"><div class="az-card">
<div class="az-title">📅 Monthly Returns — Each cell = month return %</div>
<div style="overflow-x:auto;">
<table style="border-collapse:collapse;width:100%;font-family:'Space Mono',monospace;font-size:12px;">
<thead><tr>
    <th style="padding:8px 12px;color:{MUTED};text-align:left;border-bottom:1px solid {BORDER};">Year</th>
"""
        for m in MONTHS:
            heat_html += f'<th style="padding:8px 6px;color:{MUTED};text-align:center;border-bottom:1px solid {BORDER};">{m}</th>'
        heat_html += "</tr></thead><tbody>"

        for yr in years:
            heat_html += f'<tr><td style="padding:8px 12px;color:{TXTW};font-weight:700;border-bottom:1px solid {BORDER};">{yr}</td>'
            yr_data = monthly_df[monthly_df['Year'] == yr]
            yr_dict = dict(zip(yr_data['Month'], yr_data['Return']))
            for m in MONTHS:
                if m in yr_dict:
                    r   = yr_dict[m]
                    sty = heat_color(r)
                    sign = "+" if r >= 0 else ""
                    heat_html += f'<td style="padding:5px;text-align:center;border-bottom:1px solid {BORDER};"><span style="display:inline-block;padding:4px 6px;border-radius:5px;{sty};min-width:58px;">{sign}{r:.1f}%</span></td>'
                else:
                    heat_html += f'<td style="padding:5px;text-align:center;border-bottom:1px solid {BORDER};"><span style="color:{MUTED};">—</span></td>'
            heat_html += "</tr>"

        heat_html += "</tbody></table></div></div></div>"
        st.markdown(heat_html, unsafe_allow_html=True)

        # ── Best / Worst months summary ────────────────────────
        if not monthly_df.empty:
            best_row  = monthly_df.loc[monthly_df['Return'].idxmax()]
            worst_row = monthly_df.loc[monthly_df['Return'].idxmin()]
            avg_ret   = monthly_df['Return'].mean()
            pos_months = (monthly_df['Return'] >= 0).sum()
            neg_months = (monthly_df['Return'] <  0).sum()

            st.markdown(f"""
<div class="cp"><div class="az-card">
<div class="az-title">📊 Monthly Statistics Summary</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:1px;">BEST MONTH</div>
        <div style="font-size:22px;font-weight:800;color:{UP};margin:6px 0;">+{best_row['Return']:.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};">{best_row['Month']} {best_row['Year']}</div>
    </div>
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:1px;">WORST MONTH</div>
        <div style="font-size:22px;font-weight:800;color:{DOWN};margin:6px 0;">{worst_row['Return']:.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};">{worst_row['Month']} {worst_row['Year']}</div>
    </div>
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:1px;">AVG MONTHLY</div>
        <div style="font-size:22px;font-weight:800;color:{ACCENT};margin:6px 0;">{avg_ret:+.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};">all-time average</div>
    </div>
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:1px;">WIN RATE</div>
        <div style="font-size:22px;font-weight:800;color:#60a5fa;margin:6px 0;">{pos_months/(pos_months+neg_months)*100:.0f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:12px;color:{MUTED};">{pos_months}↑ / {neg_months}↓ months</div>
    </div>
</div>
</div></div>
""", unsafe_allow_html=True)

        # ── ROW 4: Volatility ──────────────────────────────────
        st.markdown('<div class="nq-sec">Volatility Analysis · 30-Day Rolling Annualised</div>',
                    unsafe_allow_html=True)

        vol_df = get_volatility_analysis(hist_df)
        st.line_chart(vol_df.set_index('Date')['Volatility'], use_container_width=True)

        if not vol_df.empty:
            cur_vol  = float(vol_df['Volatility'].iloc[-1])
            avg_vol  = float(vol_df['Volatility'].mean())
            max_vol  = float(vol_df['Volatility'].max())
            vol_clr  = UP if cur_vol < avg_vol else "#fbb650" if cur_vol < avg_vol * 1.5 else DOWN
            vol_lbl  = "🟢 Low Volatility" if cur_vol < avg_vol else "🟡 Moderate" if cur_vol < avg_vol * 1.5 else "🔴 High Volatility"
            st.markdown(f"""
<div class="cp"><div class="az-card">
<div class="az-title">📉 Volatility Summary</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">CURRENT (30d)</div>
        <div style="font-size:24px;font-weight:800;color:{vol_clr};margin:6px 0;">{cur_vol:.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">{vol_lbl}</div>
    </div>
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">HISTORICAL AVG</div>
        <div style="font-size:24px;font-weight:800;color:{ACCENT};margin:6px 0;">{avg_vol:.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">all-time mean</div>
    </div>
    <div style="text-align:center;padding:14px;background:{BG2};border-radius:10px;border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">ALL-TIME PEAK</div>
        <div style="font-size:24px;font-weight:800;color:{DOWN};margin:6px 0;">{max_vol:.1f}%</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{MUTED};">highest ever</div>
    </div>
</div>
</div></div>
""", unsafe_allow_html=True)

        # ── ROW 5: Volume Anomalies ────────────────────────────
        st.markdown('<div class="nq-sec">Volume Anomalies · Unusual Activity (2× Average)</div>',
                    unsafe_allow_html=True)

        anomalies = get_volume_anomalies(hist_df)

        if anomalies.empty:
            st.info("No significant volume anomalies detected in the dataset.")
        else:
            anom_html = f"""
<div class="cp"><div class="az-card">
<div class="az-title">⚡ Recent Volume Spikes (last 30 anomaly days)</div>
<table style="width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:12px;">
<thead><tr>
    <th style="padding:8px 12px;color:{MUTED};text-align:left;border-bottom:1px solid {BORDER};">Date</th>
    <th style="padding:8px 12px;color:{MUTED};text-align:right;border-bottom:1px solid {BORDER};">Close Price</th>
    <th style="padding:8px 12px;color:{MUTED};text-align:right;border-bottom:1px solid {BORDER};">Volume Ratio</th>
    <th style="padding:8px 12px;color:{MUTED};text-align:center;border-bottom:1px solid {BORDER};">Signal</th>
</tr></thead><tbody>
"""
            for idx, row in anomalies.iterrows():
                date_str = str(idx)[:10]
                row_clr  = UP if "Bullish" in str(row['Direction']) else DOWN
                anom_html += f"""<tr>
    <td style="padding:8px 12px;color:{MUTED};border-bottom:1px solid {BORDER};">{date_str}</td>
    <td style="padding:8px 12px;color:{TXTW};text-align:right;border-bottom:1px solid {BORDER};">${float(row['Close']):,.2f}</td>
    <td style="padding:8px 12px;color:{ACCENT};text-align:right;border-bottom:1px solid {BORDER};">{float(row['Vol_Ratio']):.1f}×</td>
    <td style="padding:8px 12px;color:{row_clr};text-align:center;border-bottom:1px solid {BORDER};">{row['Direction']}</td>
</tr>"""

            anom_html += "</tbody></table></div></div>"
            st.markdown(anom_html, unsafe_allow_html=True)

        # ── ROW 6: FULL HISTORICAL RECORDS TABLE ──────────────
        st.markdown('<div class="nq-sec">Full Historical Record · Every Date Classified</div>',
                    unsafe_allow_html=True)

        # Build the complete enriched table from raw hist_df
        full_df = hist_df.copy()
        full_df.index = pd.to_datetime(full_df.index)

        close_s  = full_df['Close'].squeeze()
        high_s   = full_df['High'].squeeze()
        low_s    = full_df['Low'].squeeze()
        vol_s    = full_df['Volume'].squeeze()

        # Compute all indicators for every row
        full_df['RSI']        = calculate_rsi(close_s)
        full_df['EMA_20']     = close_s.ewm(span=20, adjust=False).mean()
        full_df['EMA_50']     = close_s.ewm(span=50, adjust=False).mean()
        full_df['EMA_200']    = close_s.ewm(span=200, adjust=False).mean()
        sma20_f               = close_s.rolling(20).mean()
        std20_f               = close_s.rolling(20).std()
        full_df['BB_Upper']   = sma20_f + 2 * std20_f
        full_df['BB_Lower']   = sma20_f - 2 * std20_f
        full_df['Vol_MA20']   = vol_s.rolling(20).mean()
        full_df['Return_1d']  = close_s.pct_change(1) * 100
        full_df['Return_7d']  = close_s.pct_change(7) * 100

        # Percentile thresholds for zone classification
        p10 = float(close_s.quantile(0.10))
        p25 = float(close_s.quantile(0.25))
        p75 = float(close_s.quantile(0.75))
        p90 = float(close_s.quantile(0.90))

        def classify_price(v):
            if v >= p90:  return ("ATH Zone",  UP,      "🚀")
            if v >= p75:  return ("High",       "#60a5fa","📈")
            if v >= p25:  return ("Normal",     ACCENT,  "➡")
            if v >= p10:  return ("Low",        "#fbb650","📉")
            return               ("ATL Zone",  DOWN,    "🔴")

        def classify_rsi(r):
            if pd.isna(r):  return ("—", MUTED)
            if r < 30:      return ("Oversold",   UP)
            if r > 70:      return ("Overbought", DOWN)
            return                 ("Neutral",    ACCENT)

        def classify_vol(v, ma):
            if pd.isna(ma) or ma == 0: return ("—", MUTED)
            ratio = v / ma
            if ratio > 2.0:   return ("Very High", DOWN)
            if ratio > 1.3:   return ("High",      "#fbb650")
            if ratio < 0.5:   return ("Very Low",  ACCENT)
            return                   ("Normal",    MUTED)

        def trend_vs_emas(price, e20, e50, e200):
            if pd.isna(e200): return ("—", MUTED)
            if price > e20 > e50 > e200: return ("Strong Bull", UP)
            if price > e200:             return ("Bull",        "#60a5fa")
            if price < e20 < e50 < e200: return ("Strong Bear", DOWN)
            if price < e200:             return ("Bear",        "#fbb650")
            return                              ("Sideways",   MUTED)

        full_df = full_df.dropna(subset=['Close'])
        display_df = full_df.sort_index(ascending=False)   # newest first

        # Search / filter controls
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            year_filter = st.selectbox(
                "Filter by Year",
                ["All"] + sorted(display_df.index.year.unique().tolist(), reverse=True),
                key="hist_year_filter"
            )
        with col_f2:
            zone_filter = st.selectbox(
                "Filter by Zone",
                ["All", "ATH Zone", "High", "Normal", "Low", "ATL Zone"],
                key="hist_zone_filter"
            )
        with col_f3:
            rsi_filter = st.selectbox(
                "Filter by RSI",
                ["All", "Oversold", "Neutral", "Overbought"],
                key="hist_rsi_filter"
            )

        # Apply filters
        filtered_df = display_df.copy()
        if year_filter != "All":
            filtered_df = filtered_df[filtered_df.index.year == int(year_filter)]
        if zone_filter != "All":
            filtered_df = filtered_df[
                filtered_df['Close'].apply(lambda v: classify_price(v)[0]) == zone_filter
            ]
        if rsi_filter != "All":
            filtered_df = filtered_df[
                filtered_df['RSI'].apply(lambda r: classify_rsi(r)[0]) == rsi_filter
            ]

        total_rows = len(filtered_df)
        PAGE_SIZE  = 50
        max_pages  = max(1, (total_rows - 1) // PAGE_SIZE + 1)
        page_num   = st.number_input(
            f"Page (1–{max_pages}, showing {min(PAGE_SIZE, total_rows)} of {total_rows} rows)",
            min_value=1, max_value=max_pages, value=1, step=1,
            key="hist_page"
        )
        page_df = filtered_df.iloc[(page_num - 1) * PAGE_SIZE : page_num * PAGE_SIZE]

        # Build the HTML table
        tbl = f"""
<div class="cp"><div class="az-card" style="overflow-x:auto;padding:16px 20px;">
<div class="az-title">📋 {total_rows} Records · {selected_name} · Newest First</div>
<table style="width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:11px;min-width:900px;">
<thead>
<tr style="border-bottom:2px solid {BORDER};">
  <th style="padding:8px 10px;color:{MUTED};text-align:left;">Date</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">Open</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">High</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">Low</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">Close</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:center;">Zone</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">RSI</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:center;">RSI Signal</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:center;">Trend</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">1d %</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">7d %</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:center;">Volume</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">BB Upper</th>
  <th style="padding:8px 10px;color:{MUTED};text-align:right;">BB Lower</th>
</tr>
</thead>
<tbody>
"""
        for dt, row in page_df.iterrows():
            date_str   = str(dt)[:10]
            o          = float(row['Open'])
            h          = float(row['High'])
            l          = float(row['Low'])
            c          = float(row['Close'])
            rsi_v      = row['RSI']
            e20        = row['EMA_20']
            e50        = row['EMA_50']
            e200       = row['EMA_200']
            bb_u       = row['BB_Upper']
            bb_l       = row['BB_Lower']
            vol        = float(row['Volume'])
            vol_ma     = row['Vol_MA20']
            r1d        = row['Return_1d']
            r7d        = row['Return_7d']

            z_lbl, z_clr, z_ico = classify_price(c)
            rsi_lbl, rsi_clr    = classify_rsi(rsi_v)
            tr_lbl, tr_clr      = trend_vs_emas(c, e20, e50, e200)
            vl_lbl, vl_clr      = classify_vol(vol, vol_ma)

            rsi_str  = f"{rsi_v:.1f}" if not pd.isna(rsi_v) else "—"
            r1d_str  = (f'<span style="color:{UP};">+{r1d:.2f}%</span>' if r1d >= 0
                        else f'<span style="color:{DOWN};">{r1d:.2f}%</span>') if not pd.isna(r1d) else "—"
            r7d_str  = (f'<span style="color:{UP};">+{r7d:.2f}%</span>' if r7d >= 0
                        else f'<span style="color:{DOWN};">{r7d:.2f}%</span>') if not pd.isna(r7d) else "—"
            bb_u_str = f"${bb_u:,.2f}" if not pd.isna(bb_u) else "—"
            bb_l_str = f"${bb_l:,.2f}" if not pd.isna(bb_l) else "—"

            # Row background: subtle tint based on zone
            row_bg = ""
            if z_lbl == "ATH Zone":  row_bg = f"background:rgba(0,212,170,.04);"
            elif z_lbl == "ATL Zone": row_bg = f"background:rgba(255,94,94,.04);"

            tbl += f"""
<tr style="border-bottom:1px solid {BORDER};{row_bg}transition:background .15s;"
    onmouseover="this.style.background='rgba(167,139,250,.07)'"
    onmouseout="this.style.background='{('rgba(0,212,170,.04)' if z_lbl=='ATH Zone' else 'rgba(255,94,94,.04)' if z_lbl=='ATL Zone' else '')}'">
  <td style="padding:7px 10px;color:{MUTED};">{date_str}</td>
  <td style="padding:7px 10px;color:{TXTW};text-align:right;">${o:,.2f}</td>
  <td style="padding:7px 10px;color:{UP};text-align:right;">${h:,.2f}</td>
  <td style="padding:7px 10px;color:{DOWN};text-align:right;">${l:,.2f}</td>
  <td style="padding:7px 10px;color:{TXTW};font-weight:700;text-align:right;">${c:,.2f}</td>
  <td style="padding:7px 10px;text-align:center;">
    <span style="background:rgba(0,0,0,.2);border:1px solid {z_clr};color:{z_clr};
      padding:2px 8px;border-radius:10px;font-size:10px;">{z_ico} {z_lbl}</span>
  </td>
  <td style="padding:7px 10px;color:{rsi_clr};text-align:right;">{rsi_str}</td>
  <td style="padding:7px 10px;text-align:center;">
    <span style="color:{rsi_clr};font-size:11px;">{rsi_lbl}</span>
  </td>
  <td style="padding:7px 10px;text-align:center;">
    <span style="color:{tr_clr};font-size:11px;">{tr_lbl}</span>
  </td>
  <td style="padding:7px 10px;text-align:right;">{r1d_str}</td>
  <td style="padding:7px 10px;text-align:right;">{r7d_str}</td>
  <td style="padding:7px 10px;text-align:center;">
    <span style="color:{vl_clr};font-size:10px;">{vl_lbl}</span>
  </td>
  <td style="padding:7px 10px;color:{MUTED};text-align:right;font-size:10px;">{bb_u_str}</td>
  <td style="padding:7px 10px;color:{MUTED};text-align:right;font-size:10px;">{bb_l_str}</td>
</tr>"""

        tbl += "</tbody></table></div></div>"
        st.markdown(tbl, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  CLOCK + CTA SECTION
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="nq-clock-section">
    <div class="nq-clock-wrap">
        <div class="nq-clock-face">
            <div class="nq-arc-orange"></div>
            <div class="nq-arc-blue"></div>
            <div class="nq-clock-inner">
                <div class="nq-clock-center" id="clockFace">
                    <div class="nq-hand nq-hand-h" id="hH"></div>
                    <div class="nq-hand nq-hand-m" id="hM"></div>
                    <div class="nq-hand nq-hand-s" id="hS"></div>
                    <div class="nq-clock-dot"></div>
                </div>
            </div>
        </div>
    </div>
    <div class="nq-cta-block">
        <div class="nq-cta-eye">Start your journey</div>
        <div class="nq-cta-h2">Trade Smarter.<br>Predict <em>Earlier.</em></div>
        <div class="nq-cta-sub">
            NeuralQuant uses deep LSTM networks trained on live market data
            to give you an edge before the market moves.
        </div>
        <div class="nq-cta-btns">
            <span class="btn-outline">⚡ See in Action</span>
            <span class="btn-solid">+ Run LSTM Now</span>
        </div>
    </div>
</div>

<script>
(function(){{
    function tick(){{
        var now=new Date(),
            s=now.getSeconds(),m=now.getMinutes(),h=now.getHours()%12;
        var sd=s*6,md=m*6+s*0.1,hd=h*30+m*0.5;
        var H=document.getElementById('hH'),M=document.getElementById('hM'),S=document.getElementById('hS');
        if(H) H.style.transform='rotate('+hd+'deg)';
        if(M) M.style.transform='rotate('+md+'deg)';
        if(S) S.style.transform='rotate('+sd+'deg)';
    }}
    tick(); setInterval(tick,1000);
}})();
</script>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="nq-footer">
    <div class="nq-footer-l">
        © 2026 NeuralQuant &nbsp;·&nbsp; LSTM v3.0 &nbsp;·&nbsp;
        Data: Yahoo Finance &nbsp;·&nbsp; Not financial advice
    </div>
    <div class="nq-footer-r">
        <a>Documentation</a>
        <a>API Reference</a>
        <a>Model Cards</a>
        <a>Privacy Policy</a>
        <a style="color:{ACCENT};">♥ Made with NeuralQuant</a>
    </div>
</div>
""", unsafe_allow_html=True)
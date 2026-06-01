import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="CineScope AI · Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------
# Custom CSS — IMDB-noir cinematic theme
# --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e0d0;
}
.stApp {
    background: #0a0a0a;
}
.block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1200px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0a00 0%, #0d0d0d 40%, #001a0a 100%);
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 3rem 3rem 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.01) 2px,
        rgba(255,255,255,0.01) 4px
    );
    pointer-events: none;
}
.hero-banner::after {
    content: '★★★★★';
    font-size: 5rem;
    opacity: 0.04;
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    letter-spacing: -0.1em;
    color: #f5c518;
}
.hero-eyebrow {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.35em;
    color: #f5c518;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    letter-spacing: 0.04em;
    line-height: 1;
    color: #ffffff;
    margin-bottom: 0.8rem;
}
.hero-title span {
    color: #f5c518;
}
.hero-subtitle {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #888;
    max-width: 480px;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    color: #f5c518;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── Card Panels ── */
.card {
    background: #111111;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-gold {
    border-left: 3px solid #f5c518;
}

/* ── Verdict Badge ── */
.verdict-wrap {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.6rem 2rem;
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    margin: 1rem 0;
}
.verdict-icon {
    font-size: 3.2rem;
    line-height: 1;
    flex-shrink: 0;
}
.verdict-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    color: #666;
    text-transform: uppercase;
}
.verdict-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    line-height: 1.1;
}
.verdict-positive { color: #4caf82; }
.verdict-negative { color: #e05252; }

/* ── Score Ring ── */
.score-wrap {
    text-align: center;
    padding: 1.2rem 0;
}
.score-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    line-height: 1;
    color: #f5c518;
}
.score-label {
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    color: #666;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Model Pills ── */
.model-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 2px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 0.1em;
    background: #1e1e1e;
    border: 1px solid #333;
    color: #aaa;
    margin-right: 0.5rem;
}
.model-pill-active {
    background: #f5c518;
    color: #0a0a0a;
    border-color: #f5c518;
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, #f5c518 0%, transparent 100%);
    margin: 2rem 0;
    opacity: 0.4;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e0d0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #f5c518 !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #f5c518 !important;
    box-shadow: 0 0 0 1px rgba(245,197,24,0.15) !important;
}
.stTextArea label {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.2em !important;
    color: #f5c518 !important;
    font-size: 0.8rem !important;
}
.stSelectbox > div > div {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e0d0 !important;
}
.stSelectbox label {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.2em !important;
    color: #f5c518 !important;
    font-size: 0.8rem !important;
}
.stButton button {
    background: #f5c518 !important;
    color: #0a0a0a !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.15em !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.65rem 2.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}
.stButton button:hover {
    opacity: 0.85 !important;
    background: #f5c518 !important;
}
.stDataFrame {
    border: 1px solid #222 !important;
    border-radius: 4px !important;
}
.stSuccess, .stInfo, .stWarning {
    border-radius: 3px !important;
}
.stSuccess {
    background: rgba(76,175,130,0.1) !important;
    border: 1px solid rgba(76,175,130,0.3) !important;
    color: #4caf82 !important;
}
.stWarning {
    background: rgba(245,197,24,0.08) !important;
    border: 1px solid rgba(245,197,24,0.2) !important;
    color: #f5c518 !important;
}

/* ── Subheader override ── */
h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.06em !important;
    color: #e8e0d0 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #666 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f5c518 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #333;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-top: 1px solid #1a1a1a;
    margin-top: 3rem;
}
.footer span { color: #f5c518; }
</style>
""", unsafe_allow_html=True)

# --------------------
# Load Models
# --------------------
@st.cache_resource(show_spinner=False)
def load_models():
    rnn   = tf.keras.models.load_model("rnn_model.h5")
    lstm  = tf.keras.models.load_model("lstm_model.h5")
    gru   = tf.keras.models.load_model("gru_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return rnn, lstm, gru, tok

rnn_model, lstm_model, gru_model, tokenizer = load_models()
MAX_LEN = 200

MODEL_META = {
    "SimpleRNN": {
        "icon": "⚡",
        "desc": "Fast & lightweight recurrent network",
        "color": "#6c9eea"
    },
    "LSTM": {
        "icon": "🧠",
        "desc": "Long-term memory architecture",
        "color": "#f5c518"
    },
    "GRU": {
        "icon": "🔮",
        "desc": "Gated recurrent unit balanced power",
        "color": "#4caf82"
    },
}

# --------------------
# Prediction
# --------------------
def predict_sentiment(model, text):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob   = float(model.predict(padded, verbose=0)[0][0])
    sentiment  = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else (1 - prob)
    return sentiment, confidence, prob

# --------------------
# Plotly theme helper
# --------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#888", size=12),
    margin=dict(l=20, r=20, t=45, b=20),
    xaxis=dict(gridcolor="#1e1e1e", zeroline=False, showline=False),
    yaxis=dict(gridcolor="#1e1e1e", zeroline=False, showline=False, range=[0, 1]),
    showlegend=False,
)

# ====================
# HERO
# ====================
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">Deep Learning · NLP · IMDB Reviews</div>
    <div class="hero-title">CineScope <span>AI</span></div>
    <div class="hero-subtitle">Decode the sentiment behind every review — powered by RNN, LSTM &amp; GRU architectures.</div>
</div>
""", unsafe_allow_html=True)

# ====================
# INPUT SECTION
# ====================
left_col, right_col = st.columns([3, 1], gap="large")

with left_col:
    st.markdown('<span class="section-label">📝  Your Review</span>', unsafe_allow_html=True)
    review = st.text_area(
        "ENTER YOUR MOVIE REVIEW",
        placeholder="e.g. "An absolute masterpiece — every frame drips with intention. Nolan outdoes himself once again…"",
        height=160,
        label_visibility="collapsed"
    )

with right_col:
    st.markdown('<span class="section-label">🤖  Select Model</span>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "MODEL",
        list(MODEL_META.keys()),
        format_func=lambda x: f"{MODEL_META[x]['icon']}  {x}",
        label_visibility="collapsed"
    )
    meta = MODEL_META[model_choice]
    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem; border-left: 3px solid {meta['color']};">
        <div style="font-size:1.8rem; margin-bottom:0.3rem;">{meta['icon']}</div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem; letter-spacing:0.1em; color:#ddd;">{model_choice}</div>
        <div style="font-size:0.82rem; color:#666; margin-top:0.2rem;">{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    analyze_btn = st.button("▶  ANALYZE REVIEW")

# ====================
# ANALYSIS
# ====================
if analyze_btn:
    if not review.strip():
        st.warning("⚠  Please enter a review before analyzing.")
        st.stop()

    models_map = {
        "SimpleRNN": rnn_model,
        "LSTM":      lstm_model,
        "GRU":       gru_model,
    }

    with st.spinner("Analyzing review across all models…"):
        sentiment, confidence, prob = predict_sentiment(models_map[model_choice], review)

        # Gather all comparisons
        comparison = []
        for name, model in models_map.items():
            s, c, p = predict_sentiment(model, review)
            comparison.append({
                "Model":          name,
                "Prediction":     s,
                "Confidence (%)": round(c * 100, 2),
                "Positive Prob":  round(p, 4),
                "Negative Prob":  round(1 - p, 4),
            })
        compare_df = pd.DataFrame(comparison)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Verdict ──
    is_positive = sentiment == "Positive"
    v_icon  = "🌟" if is_positive else "💀"
    v_class = "verdict-positive" if is_positive else "verdict-negative"
    stars   = int(round(confidence * 5))
    star_str = "★" * stars + "☆" * (5 - stars)

    st.markdown(f"""
    <div class="verdict-wrap">
        <div class="verdict-icon">{v_icon}</div>
        <div>
            <div class="verdict-label">Selected Model Verdict — {model_choice}</div>
            <div class="verdict-value {v_class}">{sentiment.upper()}</div>
            <div style="color:#f5c518; font-size:1.1rem; letter-spacing:0.05em; margin-top:0.2rem;">{star_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Confidence", f"{confidence*100:.1f}%")
    k2.metric("Positive Prob", f"{prob:.4f}")
    k3.metric("Negative Prob", f"{1-prob:.4f}")
    word_count = len(review.split())
    k4.metric("Words Analyzed", str(word_count))

    st.markdown("")

    # ── Side-by-side charts ──
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<span class="section-label">📊  Sentiment Probability</span>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Class":       ["Negative", "Positive"],
            "Probability": [round(1 - prob, 4), round(prob, 4)],
            "Color":       ["#e05252", "#4caf82"],
        })
        fig1 = go.Figure(go.Bar(
            x=prob_df["Class"],
            y=prob_df["Probability"],
            marker_color=["#e05252", "#4caf82"],
            marker_line_width=0,
            text=[f"{v:.1%}" for v in prob_df["Probability"]],
            textposition="outside",
            textfont=dict(family="Bebas Neue", size=14, color="#ddd"),
            width=0.45,
        ))
        fig1.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Pos vs Neg — " + model_choice, font=dict(family="Bebas Neue", size=14, color="#666")),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown('<span class="section-label">🏆  Model Confidence Comparison</span>', unsafe_allow_html=True)
        colors = [MODEL_META[m]["color"] for m in compare_df["Model"]]
        fig2 = go.Figure(go.Bar(
            x=compare_df["Model"],
            y=compare_df["Confidence (%)"] / 100,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in compare_df["Confidence (%)"]],
            textposition="outside",
            textfont=dict(family="Bebas Neue", size=14, color="#ddd"),
            width=0.45,
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Confidence Across All Models", font=dict(family="Bebas Neue", size=14, color="#666")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar / Polar chart ──
    st.markdown('<span class="section-label">🕸  Positive Probability Radar</span>', unsafe_allow_html=True)
    labels  = compare_df["Model"].tolist()
    values  = compare_df["Positive Prob"].tolist()
    fig3 = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        line_color="#f5c518",
        fillcolor="rgba(245,197,24,0.12)",
        marker=dict(color="#f5c518", size=8),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#888", size=13),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(linecolor="#2a2a2a", gridcolor="#1a1a1a", tickfont=dict(family="Bebas Neue", size=13, color="#ccc")),
            radialaxis=dict(linecolor="#2a2a2a", gridcolor="#1a1a1a", tickfont=dict(size=10, color="#555"), range=[0, 1]),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Model Comparison Table ──
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">📋  Full Model Comparison</span>', unsafe_allow_html=True)

    def style_pred(val):
        if val == "Positive":
            return "color: #4caf82; font-weight:600;"
        return "color: #e05252; font-weight:600;"

    styled = compare_df.style \
        .applymap(style_pred, subset=["Prediction"]) \
        .format({"Confidence (%)": "{:.2f}%", "Positive Prob": "{:.4f}", "Negative Prob": "{:.4f}"}) \
        .set_properties(**{"background-color": "#111", "color": "#e8e0d0", "border": "1px solid #222"}) \
        .set_table_styles([
            {"selector": "th", "props": [("background", "#1a1a1a"), ("color", "#f5c518"), ("font-family", "DM Sans"), ("font-size", "0.75rem"), ("letter-spacing", "0.1em"), ("text-transform", "uppercase")]},
            {"selector": "tr:hover td", "props": [("background-color", "#161616")]},
        ])

    st.dataframe(compare_df, use_container_width=True)

    # ── Review Echo ──
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">💬  Reviewed Text</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card card-gold">
        <div style="font-family:'Playfair Display',serif; font-style:italic; font-size:1.05rem; color:#b0a890; line-height:1.7;">
            "{review[:800]}{"…" if len(review) > 800 else ""}"
        </div>
        <div style="margin-top:1rem; font-size:0.78rem; color:#444; letter-spacing:0.1em; text-transform:uppercase;">
            {word_count} words &nbsp;·&nbsp; analyzed by {model_choice} &nbsp;·&nbsp;
            <span style="color:{'#4caf82' if is_positive else '#e05252'};">{sentiment}</span>
            &nbsp;with {confidence*100:.1f}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================
# EMPTY STATE
# ====================
else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem; color:#2a2a2a;">
        <div style="font-size:5rem; margin-bottom:1rem; opacity:0.4;">🎬</div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:0.2em; color:#2a2a2a;">
            ENTER A REVIEW TO BEGIN
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================
# FOOTER
# ====================
st.markdown("""
<div class="footer">
    CineScope AI &nbsp;·&nbsp; Powered by <span>TensorFlow</span> &nbsp;·&nbsp;
    RNN · LSTM · GRU &nbsp;·&nbsp; IMDB Sentiment Analysis
</div>
""", unsafe_allow_html=True)

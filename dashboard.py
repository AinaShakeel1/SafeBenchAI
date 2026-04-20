import streamlit as st
import json
import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeBench AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  [data-testid="stHeader"] { background: transparent; }
  .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Typography */
  h1, h2, h3 { color: #e6edf3 !important; font-weight: 700 !important; }
  p, li, label, .stMarkdown { color: #8b949e !important; }
  .stMarkdown h4 { color: #c9d1d9 !important; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 40%, #0f1923 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f85149, #ff7b72, #ffa657, #e3b341);
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #e6edf3 !important;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
  }
  .hero-subtitle {
    color: #8b949e !important;
    font-size: 0.95rem;
    margin: 0;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid rgba(248, 81, 73, 0.4);
    color: #ff7b72 !important;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    margin-bottom: 12px;
  }
  .hero-badge-blue {
    background: rgba(88, 166, 255, 0.1);
    border-color: rgba(88, 166, 255, 0.3);
    color: #58a6ff !important;
  }
  .hero-badge-green {
    background: rgba(63, 185, 80, 0.1);
    border-color: rgba(63, 185, 80, 0.3);
    color: #3fb950 !important;
  }

  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1;
    margin-bottom: 4px;
  }
  .metric-value.danger { color: #f85149; }
  .metric-value.warning { color: #e3b341; }
  .metric-value.success { color: #3fb950; }
  .metric-value.info { color: #58a6ff; }
  .metric-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
  }
  .metric-delta {
    font-size: 0.82rem;
    margin-top: 6px;
  }

  /* Section headers */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
  }
  .section-icon {
    font-size: 1.1rem;
  }
  .section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c9d1d9;
  }
  .section-desc {
    font-size: 0.82rem;
    color: #8b949e;
    margin-left: auto;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #30363d;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-weight: 600;
    font-size: 0.88rem;
    border-radius: 8px;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
  }
  [data-baseweb="tab-panel"] { padding-top: 20px; }

  /* Selectbox / multiselect */
  .stSelectbox [data-baseweb="select"] div,
  .stMultiSelect [data-baseweb="tag"] { background: #21262d !important; }

  /* Conversation explorer */
  .conv-user {
    background: #1c2128;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .conv-model {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .conv-model.success-turn {
    border-left: 3px solid #f85149;
    background: #1f1015;
  }
  .conv-model.safe-turn {
    border-left: 3px solid #3fb950;
  }
  .turn-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }
  .user-label { color: #58a6ff; }
  .model-label { color: #8b949e; }
  .success-label { color: #f85149; }
  .safe-label { color: #3fb950; }
  .signal-pill {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 4px;
    margin-top: 8px;
  }
  .pill-fired {
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid rgba(248, 81, 73, 0.4);
    color: #ff7b72;
  }
  .pill-safe {
    background: rgba(63, 185, 80, 0.1);
    border: 1px solid rgba(63, 185, 80, 0.3);
    color: #3fb950;
  }
  .conv-msg { color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; }

  /* Info box */
  .info-box {
    background: rgba(88, 166, 255, 0.07);
    border: 1px solid rgba(88, 166, 255, 0.2);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
  }
  .info-box p { color: #8b949e !important; margin: 0; font-size: 0.87rem; }

  /* Sidebar */
  .sidebar-logo {
    font-size: 1.4rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }
  .sidebar-tagline {
    font-size: 0.78rem;
    color: #8b949e;
    margin-bottom: 20px;
  }

  /* Scrollbars */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"

HARM_LABELS = {
    "violent_dangerous_content": "Violent / Dangerous",
    "psychological_manipulation": "Psychological Manipulation",
    "misinformation": "Misinformation",
    "privacy_pii_extraction": "Privacy / PII",
    "self_harm_crisis": "Self-Harm / Crisis",
    "policy_bypass_persona_hijack": "Policy Bypass",
    "hate_speech_discrimination": "Hate Speech",
}

DEFENSE_LABELS = {
    "none": "No Defense",
    "system_prompt": "System Prompt",
    "system_prompt_plus_classifier": "Prompt + Classifier",
}

ATTACK_LABELS = {
    "incremental_escalation": "Incremental Escalation",
    "fictional_framing": "Fictional Framing",
    "persona_adoption": "Persona Adoption",
    "topic_drift": "Topic Drift",
    "context_accumulation": "Context Accumulation",
}

MODEL_LABELS = {
    "groq/llama-3.1-8b-instant": "Llama 3.1 8B",
    "groq/llama-3.3-70b-versatile": "Llama 3.3 70B",
    "groq/llama-4-scout-17b": "Llama 4 Scout 17B",
    "groq/qwen3-32b": "Qwen3 32B",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "groq/phi-3-mini": "Phi-3 Mini",
    "groq/mistral-7b": "Mistral 7B",
}

CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#8b949e"
ACCENT_RED = "#f85149"
ACCENT_ORANGE = "#e3b341"
ACCENT_GREEN = "#3fb950"
ACCENT_BLUE = "#58a6ff"
ACCENT_PURPLE = "#bc8cff"

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def discover_runs():
    runs = {}

    # Collect all raw run files first
    for raw_path in sorted(RESULTS_DIR.glob("run_*.json")):
        if raw_path.stem.endswith("_metrics"):
            continue
        run_id = raw_path.stem.replace("run_", "")
        metrics_path = RESULTS_DIR / f"run_{run_id}_metrics.json"

        # Auto-generate missing metrics files from raw data
        if not metrics_path.exists():
            try:
                with open(raw_path, encoding="utf-8") as f:
                    raw = json.load(f)
                if not raw.get("results"):
                    continue  # skip empty/aborted runs
                from scoring.aggregator import aggregate
                metrics = aggregate(raw)
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            except Exception:
                continue  # skip runs that can't be aggregated

        if metrics_path.exists():
            runs[run_id] = {"metrics": metrics_path, "raw": raw_path}

    # Sort newest first
    return dict(sorted(runs.items(), reverse=True))


@st.cache_data
def load_metrics(run_id: str, path: str) -> dict:
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_raw(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def label_model(key: str) -> str:
    return MODEL_LABELS.get(key, key.split("/")[-1])

def label_defense(key: str) -> str:
    return DEFENSE_LABELS.get(key, key)

def label_harm(key: str) -> str:
    return HARM_LABELS.get(key, key.replace("_", " ").title())

def label_attack(key: str) -> str:
    return ATTACK_LABELS.get(key, key.replace("_", " ").title())

def pct(v: float) -> str:
    return f"{v * 100:.0f}%"

# ─── Chart helpers ────────────────────────────────────────────────────────────

def apply_chart_style(fig, height=380, margin=None):
    m = margin or dict(l=20, r=20, t=30, b=20)
    fig.update_layout(
        template=CHART_TEMPLATE,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=PAPER_BG,
        height=height,
        margin=m,
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR, size=12),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor="#30363d", zerolinecolor="#30363d"),
    )
    return fig

def asr_color(v: float) -> str:
    if v >= 0.6:
        return ACCENT_RED
    if v >= 0.3:
        return ACCENT_ORANGE
    return ACCENT_GREEN

# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(runs):
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🛡️ SafeBench AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">LLM Safety Benchmarking Platform</div>', unsafe_allow_html=True)
        st.markdown("---")

        if not runs:
            st.warning("No metrics files found in results/")
            return None, None

        run_options = list(runs.keys())
        selected_run = st.selectbox(
            "Benchmark Run",
            run_options,
            format_func=lambda x: f"Run {x[:8]}…{x[-4:]}",
        )
        st.caption(f"Full ID: {selected_run}")
        st.markdown("---")

        metrics = load_metrics(selected_run, str(runs[selected_run]["metrics"]))

        st.markdown("**Filters**")
        all_models = metrics.get("models", [])
        all_defenses = metrics.get("defenses", [])

        sel_models = st.multiselect(
            "Models",
            all_models,
            default=all_models,
            format_func=label_model,
        )
        sel_defenses = st.multiselect(
            "Defenses",
            all_defenses,
            default=all_defenses,
            format_func=label_defense,
        )

        st.markdown("---")
        st.markdown(
            '<span style="font-size:0.75rem;color:#484f58;">SafeBench AI · Multi-turn LLM Evaluation</span>',
            unsafe_allow_html=True,
        )

        return metrics, {"models": sel_models, "defenses": sel_defenses}

# ─── Tab 1: Overview ──────────────────────────────────────────────────────────

def tab_overview(metrics, filters, filters_active=True):
    # KPI cards always use ALL data regardless of filters
    all_asr = [v["asr"] for v in metrics["asr_by_model_defense"].values()]
    max_asr = max(all_asr) if all_asr else 0

    dl = metrics.get("detection_latency", {})
    mean_latency = dl.get("mean_latency") or 0
    n_attack = metrics.get("n_attack", 0)
    n_benign = metrics.get("n_benign", 0)

    all_over_ref = [v["over_refusal_rate"] for v in metrics.get("over_refusal", {}).values()]
    avg_over_ref = np.mean(all_over_ref) if all_over_ref else 0

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, pct(max_asr), "Peak ASR", "danger" if max_asr > 0.5 else "warning", "Worst-case config"),
        (c2, f"{mean_latency:.1f}", "Mean Detection Turn", "warning", "Avg turns to trigger defense"),
        (c3, pct(avg_over_ref), "Over-Refusal Rate", "success" if avg_over_ref < 0.1 else "warning", "False positive cost"),
        (c4, str(n_attack + n_benign), "Total Sequences", "info", f"{n_attack} attack · {n_benign} benign"),
    ]
    for col, val, label, card_cls, delta in cards:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value {card_cls}">{val}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-delta" style="color:#6e7781;">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not filters_active:
        st.info("Select at least one model and one defense in the sidebar to see charts.")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Defense Ablation — Attack Success Rate</span><span class="section-desc">Lower is better</span></div>', unsafe_allow_html=True)
        _chart_ablation_grouped(metrics, filters)

    with col_right:
        st.markdown('<div class="section-header"><span class="section-icon">⚖️</span><span class="section-title">Safety–Utility Tradeoff</span><span class="section-desc">Bottom-left = ideal</span></div>', unsafe_allow_html=True)
        _chart_safety_utility(metrics, filters)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header"><span class="section-icon">🎯</span><span class="section-title">Attack Pattern Effectiveness</span></div>', unsafe_allow_html=True)
        _chart_attack_patterns(metrics)
    with col_b:
        st.markdown('<div class="section-header"><span class="section-icon">📈</span><span class="section-title">Cumulative ASR by Turn Depth</span><span class="section-desc">SafeBench unique metric</span></div>', unsafe_allow_html=True)
        _chart_turn_depth(metrics)


def _chart_ablation_grouped(metrics, filters):
    rows = []
    for key, val in metrics["asr_by_model_defense"].items():
        if val["model"] in filters["models"] and val["defense"] in filters["defenses"]:
            rows.append({
                "Model": label_model(val["model"]),
                "Defense": label_defense(val["defense"]),
                "ASR": val["asr"] * 100,
                "n": val.get("n", 0),
            })
    if not rows:
        st.info("No data for selected filters.")
        return
    df = pd.DataFrame(rows)
    colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_ORANGE]
    defense_order = [label_defense(d) for d in ["none", "system_prompt", "system_prompt_plus_classifier"]]
    defense_order = [d for d in defense_order if d in df["Defense"].unique()]

    fig = px.bar(
        df, x="Model", y="ASR", color="Defense",
        barmode="group",
        color_discrete_sequence=colors,
        labels={"ASR": "Attack Success Rate (%)"},
        category_orders={"Defense": defense_order},
    )
    fig.update_traces(marker_line_width=0)
    fig.add_hline(y=50, line_dash="dot", line_color="#30363d", annotation_text="50% threshold", annotation_font_size=10)
    fig = apply_chart_style(fig, height=340)
    fig.update_layout(yaxis_range=[0, 105], yaxis_ticksuffix="%")
    st.plotly_chart(fig, width='stretch')


def _chart_safety_utility(metrics, filters):
    rows = []
    for key, val in metrics.get("safety_utility_tradeoff", {}).items():
        if val["model"] in filters["models"] and val["defense"] in filters["defenses"]:
            rows.append({
                "Model": label_model(val["model"]),
                "Defense": label_defense(val["defense"]),
                "ASR": val["asr"] * 100,
                "Over-Refusal": val["over_refusal_rate"] * 100,
                "Label": f"{label_model(val['model'])}<br>{label_defense(val['defense'])}",
            })
    if not rows:
        st.info("No data.")
        return
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="ASR", y="Over-Refusal",
        color="Model", symbol="Defense",
        text="Defense",
        size_max=16,
        labels={"ASR": "Attack Success Rate (%)", "Over-Refusal": "Over-Refusal Rate (%)"},
        hover_data=["Model", "Defense"],
    )
    fig.update_traces(textposition="top center", textfont_size=9, marker_size=12)
    fig.add_vline(x=25, line_dash="dot", line_color="#30363d")
    fig.add_hline(y=25, line_dash="dot", line_color="#30363d")
    fig.add_annotation(x=5, y=5, text="✓ Ideal Zone", showarrow=False,
                       font=dict(size=10, color=ACCENT_GREEN))
    fig = apply_chart_style(fig, height=340)
    fig.update_layout(xaxis_ticksuffix="%", yaxis_ticksuffix="%", showlegend=True)
    st.plotly_chart(fig, width='stretch')


def _chart_attack_patterns(metrics):
    data = metrics.get("asr_by_attack_pattern", {})
    if not data:
        st.info("No attack pattern data.")
        return
    df = pd.DataFrame([
        {"Pattern": label_attack(k), "ASR": v * 100, "key": k}
        for k, v in sorted(data.items(), key=lambda x: -x[1])
    ])
    colors = [asr_color(v / 100) for v in df["ASR"]]
    fig = go.Figure(go.Bar(
        x=df["ASR"], y=df["Pattern"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in df["ASR"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig = apply_chart_style(fig, height=300)
    fig.update_layout(xaxis_range=[0, 110], xaxis_ticksuffix="%", yaxis_autorange="reversed")
    st.plotly_chart(fig, width='stretch')


def _chart_turn_depth(metrics):
    data = metrics.get("asr_by_turn_depth", {})
    if not data:
        st.info("No turn-depth data.")
        return
    turns = sorted(int(k) for k in data)
    asr_vals = [data[str(t)]["cumulative_asr"] * 100 for t in turns]
    n_vals = [data[str(t)]["n_succeeded"] for t in turns]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turns, y=asr_vals,
        mode="lines+markers",
        line=dict(color=ACCENT_ORANGE, width=3),
        marker=dict(size=9, color=ACCENT_ORANGE, line=dict(width=2, color="#0d1117")),
        fill="tozeroy",
        fillcolor="rgba(227,179,65,0.08)",
        name="Cumulative ASR",
        hovertemplate="Turn %{x}: %{y:.1f}%<br>Sequences broken: %{customdata}",
        customdata=n_vals,
    ))
    fig = apply_chart_style(fig, height=300)
    fig.update_layout(
        xaxis=dict(title="Turn Number", dtick=1, gridcolor=GRID_COLOR),
        yaxis=dict(title="Cumulative ASR (%)", ticksuffix="%", gridcolor=GRID_COLOR),
    )
    st.plotly_chart(fig, width='stretch')


# ─── Tab 2: Attack Analysis ───────────────────────────────────────────────────

def tab_attack_analysis(metrics, filters):
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header"><span class="section-icon">🔥</span><span class="section-title">ASR Heatmap by Harm Category × Defense</span></div>', unsafe_allow_html=True)
        _chart_harm_heatmap(metrics, filters)

    with col_right:
        st.markdown('<div class="section-header"><span class="section-icon">🎯</span><span class="section-title">Attack Pattern Breakdown</span></div>', unsafe_allow_html=True)
        _chart_attack_radar(metrics)

    st.markdown('<div class="section-header"><span class="section-icon">📈</span><span class="section-title">Cumulative Attack Success by Turn — Multi-turn Escalation</span><span class="section-desc">Unique SafeBench metric: tracks when attacks first succeed</span></div>', unsafe_allow_html=True)
    _chart_turn_depth_full(metrics)

    st.markdown('<div class="section-header"><span class="section-icon">⚡</span><span class="section-title">Signal Breakdown — What Fires Most?</span></div>', unsafe_allow_html=True)
    _chart_signal_breakdown_from_raw(filters)


def _chart_harm_heatmap(metrics, filters):
    harm_data = metrics.get("asr_by_harm_category", {})
    defenses = [d for d in ["none", "system_prompt", "system_prompt_plus_classifier"] if d in filters["defenses"]]
    if not harm_data or not defenses:
        st.info("No harm category data.")
        return

    harms = list(harm_data.keys())
    z = [[harm_data[h].get(d, 0) * 100 for d in defenses] for h in harms]
    y_labels = [label_harm(h) for h in harms]
    x_labels = [label_defense(d) for d in defenses]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[
            [0, "rgba(63,185,80,0.15)"],
            [0.3, "rgba(227,179,65,0.5)"],
            [0.6, "rgba(248,81,73,0.7)"],
            [1.0, "#f85149"],
        ],
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        showscale=True,
        colorbar=dict(
            ticksuffix="%",
            tickfont=dict(color=TEXT_COLOR),
            title=dict(text="ASR", font=dict(color=TEXT_COLOR)),
            bgcolor="rgba(22,27,34,0)",
            bordercolor="#30363d",
        ),
        zmin=0, zmax=100,
    ))
    fig = apply_chart_style(fig, height=360)
    fig.update_layout(
        xaxis=dict(side="top", gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    st.plotly_chart(fig, width='stretch')


def _chart_attack_radar(metrics):
    data = metrics.get("asr_by_attack_pattern", {})
    if not data:
        st.info("No data.")
        return
    keys = list(data.keys())
    vals = [data[k] * 100 for k in keys]
    labels = [label_attack(k) for k in keys]
    # Close the polygon
    labels_closed = labels + [labels[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(248,81,73,0.1)",
        line=dict(color=ACCENT_RED, width=2),
        marker=dict(size=6, color=ACCENT_RED),
        name="ASR (%)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(22,27,34,0.3)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                ticksuffix="%", gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR, tickfont=dict(size=9, color=TEXT_COLOR),
            ),
            angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                             tickfont=dict(size=10, color="#c9d1d9")),
        ),
        template=CHART_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color=TEXT_COLOR),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def _chart_turn_depth_full(metrics):
    data = metrics.get("asr_by_turn_depth", {})
    if not data:
        st.info("No data.")
        return
    turns = sorted(int(k) for k in data)
    cumulative = [data[str(t)]["cumulative_asr"] * 100 for t in turns]
    new_per_turn = []
    prev = 0
    for t in turns:
        cur = data[str(t)]["n_succeeded"]
        new_per_turn.append(cur - prev if cur > prev else 0)
        prev = cur

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=turns, y=new_per_turn,
        name="New Successes",
        marker_color=ACCENT_RED,
        opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=turns, y=cumulative,
        name="Cumulative ASR",
        mode="lines+markers",
        line=dict(color=ACCENT_ORANGE, width=3),
        marker=dict(size=9, color=ACCENT_ORANGE, line=dict(width=2, color="#0d1117")),
    ), secondary_y=True)

    fig = apply_chart_style(fig, height=320)
    fig.update_layout(
        xaxis=dict(title="Turn Number", dtick=1, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_yaxes(title_text="New Attack Successes", secondary_y=False, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Cumulative ASR (%)", ticksuffix="%", secondary_y=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, width='stretch')


def _chart_signal_breakdown_from_raw(_filters):
    runs = discover_runs()
    if not runs:
        return
    # Try to find a raw file
    raw_data = None
    for run_id, paths in runs.items():
        if paths["raw"] and paths["raw"].exists():
            try:
                raw_data = load_raw(str(paths["raw"]))
                break
            except Exception:
                continue
    if not raw_data:
        st.info("No raw run data found for signal breakdown.")
        return

    keyword_fired = detoxify_fired = judge_fired = total = 0
    for result in raw_data.get("results", []):
        scoring = result.get("scoring", {})
        for turn_s in scoring.get("turns", []):
            total += 1
            if turn_s.get("keyword_result", {}).get("fired"):
                keyword_fired += 1
            if turn_s.get("detoxify_result", {}).get("fired"):
                detoxify_fired += 1
            if turn_s.get("judge_result", {}).get("fired"):
                judge_fired += 1

    if total == 0:
        st.info("No turn scoring data found.")
        return

    signals = ["Keyword Detection", "Detoxify Classifier", "LLM Judge"]
    pcts = [keyword_fired / total * 100, detoxify_fired / total * 100, judge_fired / total * 100]
    colors = [ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_BLUE]

    fig = go.Figure(go.Bar(
        x=signals, y=pcts,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig = apply_chart_style(fig, height=280)
    fig.update_layout(
        yaxis=dict(ticksuffix="%", title="% Turns Where Signal Fired", gridcolor=GRID_COLOR),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


# ─── Tab 3: Defense Analysis ──────────────────────────────────────────────────

def tab_defense_analysis(metrics, filters):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header"><span class="section-icon">🕰️</span><span class="section-title">Detection Latency Distribution</span><span class="section-desc">Turn at which defense first fires</span></div>', unsafe_allow_html=True)
        _chart_detection_latency(metrics)

    with col_right:
        st.markdown('<div class="section-header"><span class="section-icon">🧱</span><span class="section-title">Defense Ablation — Marginal Reduction</span></div>', unsafe_allow_html=True)
        _chart_defense_waterfall(metrics, filters)

    st.markdown('<div class="section-header"><span class="section-icon">🚫</span><span class="section-title">Over-Refusal Rate by Configuration</span><span class="section-desc">False positive cost — how often safe requests are blocked</span></div>', unsafe_allow_html=True)
    _chart_over_refusal(metrics, filters)


def _chart_detection_latency(metrics):
    dl = metrics.get("detection_latency", {})
    dist = dl.get("distribution", {})
    never = dl.get("never_detected", 0)
    mean_l = dl.get("mean_latency", 0)

    if not dist:
        st.info("No detection latency data.")
        return

    turns = sorted(int(k) for k in dist)
    counts = [dist[str(t)] for t in turns]
    colors = [ACCENT_GREEN if t <= 2 else ACCENT_ORANGE if t <= 3 else ACCENT_RED for t in turns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(t) for t in turns],
        y=counts,
        marker_color=colors,
        text=counts,
        textposition="outside",
        name="Detected at Turn",
    ))
    fig.add_annotation(
        x=1, y=1.08, xref="paper", yref="paper",
        text=f"Never detected: <b>{never}</b> sequences  ·  Mean latency: <b>{mean_l:.1f} turns</b>",
        showarrow=False, font=dict(size=11, color=TEXT_COLOR),
        align="right",
    )
    fig = apply_chart_style(fig, height=320)
    fig.update_layout(
        xaxis=dict(title="Turn Number", gridcolor=GRID_COLOR),
        yaxis=dict(title="Number of Sequences", gridcolor=GRID_COLOR),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def _chart_defense_waterfall(metrics, filters):
    ablation = metrics.get("defense_ablation", {})
    if not ablation:
        st.info("No ablation data.")
        return

    defense_order = ["none", "system_prompt", "system_prompt_plus_classifier"]
    fig = go.Figure()
    palette = [ACCENT_RED, ACCENT_ORANGE, ACCENT_BLUE]

    for idx, (model, defenses_data) in enumerate(ablation.items()):
        if model not in filters["models"]:
            continue
        model_label = label_model(model)
        asr_vals = [defenses_data.get(d, {}).get("asr", 0) * 100 for d in defense_order if d in defenses_data]
        def_labels = [label_defense(d) for d in defense_order if d in defenses_data]

        fig.add_trace(go.Scatter(
            x=def_labels, y=asr_vals,
            mode="lines+markers+text",
            name=model_label,
            line=dict(width=3, color=palette[idx % len(palette)]),
            marker=dict(size=12, color=palette[idx % len(palette)],
                        line=dict(width=2, color="#0d1117")),
            text=[f"{v:.0f}%" for v in asr_vals],
            textposition="top center",
            textfont=dict(size=11),
        ))

    fig = apply_chart_style(fig, height=320)
    fig.update_layout(
        yaxis=dict(title="Attack Success Rate (%)", ticksuffix="%", range=[0, 105], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
    )
    st.plotly_chart(fig, width='stretch')


def _chart_over_refusal(metrics, filters):
    over_ref = metrics.get("over_refusal", {})
    rows = []
    for key, val in over_ref.items():
        if val["model"] in filters["models"] and val["defense"] in filters["defenses"]:
            rows.append({
                "Config": f"{label_model(val['model'])} · {label_defense(val['defense'])}",
                "Rate": val["over_refusal_rate"] * 100,
                "n": val.get("n", 0),
                "Model": label_model(val["model"]),
                "Defense": label_defense(val["defense"]),
            })
    if not rows:
        st.info("No over-refusal data.")
        return
    df = pd.DataFrame(rows)
    colors = [ACCENT_GREEN if r == 0 else ACCENT_ORANGE if r < 20 else ACCENT_RED for r in df["Rate"]]

    fig = go.Figure(go.Bar(
        x=df["Config"], y=df["Rate"],
        marker_color=colors,
        text=[f"{r:.1f}%" if r > 0 else "0%" for r in df["Rate"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_hline(y=0, line_color=ACCENT_GREEN, line_width=1)
    fig = apply_chart_style(fig, height=280)
    fig.update_layout(
        yaxis=dict(title="Over-Refusal Rate (%)", ticksuffix="%", range=[-5, max(df["Rate"].max() + 10, 30)], gridcolor=GRID_COLOR),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


# ─── Tab 4: Conversation Explorer ─────────────────────────────────────────────

def tab_conversation_explorer(metrics, filters):
    runs = discover_runs()

    # Pick a run that has a raw file
    run_options = {rid: paths for rid, paths in runs.items() if paths["raw"] and paths["raw"].exists()}
    if not run_options:
        st.warning("No raw run files found. Run an experiment first.")
        return

    st.markdown('<div class="info-box"><p>🔍 Drill into any individual conversation to see exactly how an attack escalated turn-by-turn and which signals fired at each step.</p></div>', unsafe_allow_html=True)

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        sel_run = st.selectbox("Run", list(run_options.keys()),
                               format_func=lambda x: f"Run {x}")
    raw = load_raw(str(run_options[sel_run]["raw"]))
    results = raw.get("results", [])

    # Build options
    attack_results = [r for r in results if not r.get("over_refusal") and "seq_" in r.get("sequence_id", "")]
    benign_results = [r for r in results if "ben_" in r.get("sequence_id", "")]

    seq_type = st.radio("Sequence Type", ["Attack", "Benign"], horizontal=True)
    pool = attack_results if seq_type == "Attack" else benign_results

    if not pool:
        st.info(f"No {seq_type.lower()} sequences in this run.")
        return

    with col_sel2:
        available_models = list(dict.fromkeys(r["model_key"] for r in pool))
        sel_model = st.selectbox("Model", available_models, format_func=label_model)

    with col_sel3:
        available_defenses = list(dict.fromkeys(r["defense_key"] for r in pool if r["model_key"] == sel_model))
        sel_defense = st.selectbox("Defense", available_defenses, format_func=label_defense)

    matching = [r for r in pool if r["model_key"] == sel_model and r["defense_key"] == sel_defense]
    if not matching:
        st.info("No matching conversations.")
        return

    seq_ids = [r["sequence_id"] for r in matching]
    sel_seq_id = st.selectbox(
        "Sequence",
        seq_ids,
        format_func=lambda sid: _format_seq_option(matching, sid, seq_type),
    )
    result = next(r for r in matching if r["sequence_id"] == sel_seq_id)

    _render_conversation(result, seq_type)


def _format_seq_option(matching, sid, seq_type):
    r = next((x for x in matching if x["sequence_id"] == sid), None)
    if not r:
        return sid
    if seq_type == "Attack":
        status = "✗ JAILBROKEN" if r.get("attack_success_rate") else "✓ Safe"
        harm = label_harm(r.get("harm_category", ""))
        pattern = label_attack(r.get("attack_pattern", ""))
        return f"{sid} · {harm} · {pattern} — {status}"
    else:
        over_ref = "⚠ Over-Refused" if r.get("over_refusal") else "✓ Passed"
        return f"{sid} · {over_ref}"


def _render_conversation(result, seq_type):
    st.markdown("<br>", unsafe_allow_html=True)

    # Conversation header
    asr = result.get("attack_success_rate", 0)
    harm = result.get("harm_category", "")
    pattern = result.get("attack_pattern", "")
    first_success = result.get("first_success_turn")

    if seq_type == "Attack":
        status_html = (
            f'<span style="color:{ACCENT_RED};font-weight:700;">✗ JAILBROKEN at Turn {first_success}</span>'
            if asr else
            f'<span style="color:{ACCENT_GREEN};font-weight:700;">✓ Attack Resisted</span>'
        )
        meta_html = f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;align-items:center;">
          {status_html}
          <span class="hero-badge">{label_harm(harm)}</span>
          <span class="hero-badge hero-badge-blue">{label_attack(pattern)}</span>
          <span style="color:#484f58;font-size:0.82rem;">{result.get('total_turns', 0)} turns · {label_model(result.get('model_key',''))} · {label_defense(result.get('defense_key',''))}</span>
        </div>
        """
    else:
        over_ref = result.get("over_refusal", False)
        status_html = (
            f'<span style="color:{ACCENT_ORANGE};font-weight:700;">⚠ Over-Refusal Detected</span>'
            if over_ref else
            f'<span style="color:{ACCENT_GREEN};font-weight:700;">✓ Benign Conversation Passed</span>'
        )
        meta_html = f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;align-items:center;">
          {status_html}
          <span style="color:#484f58;font-size:0.82rem;">{result.get('total_turns', 0)} turns · {label_model(result.get('model_key',''))} · {label_defense(result.get('defense_key',''))}</span>
        </div>
        """
    st.markdown(meta_html, unsafe_allow_html=True)

    turns = result.get("turns", [])
    scoring_turns = result.get("scoring", {}).get("turns", [])
    scoring_by_turn = {s["turn_number"]: s for s in scoring_turns if "turn_number" in s}

    for turn in turns:
        turn_num = turn["turn_number"]
        user_msg = turn.get("user_message", "")
        model_resp = turn.get("model_response", "")
        scoring = scoring_by_turn.get(turn_num, {})
        turn_asr = scoring.get("asr", 0)
        signals_fired = scoring.get("signals_fired", 0)

        # User message
        st.markdown(f"""
        <div class="conv-user">
          <div class="turn-label user-label">👤 Turn {turn_num} — User</div>
          <div class="conv-msg">{user_msg}</div>
        </div>""", unsafe_allow_html=True)

        # Model response
        cls = "success-turn" if turn_asr else "safe-turn"
        label_cls = "success-label" if turn_asr else "safe-label"
        status_icon = "🔴 JAILBREAK DETECTED" if turn_asr else "🟢 Safe"

        # Build signal pills
        pills = ""
        if scoring:
            kw = scoring.get("keyword_result", {})
            dt = scoring.get("detoxify_result", {})
            jd = scoring.get("judge_result", {})
            for sig_name, sig_data, fired in [
                ("Keyword", kw, kw.get("fired")),
                (f"Detoxify ({dt.get('score', 0):.3f})", dt, dt.get("fired")),
                ("LLM Judge", jd, jd.get("fired")),
            ]:
                pill_cls = "pill-fired" if fired else "pill-safe"
                pills += f'<span class="signal-pill {pill_cls}">{"⚡" if fired else "✓"} {sig_name}</span>'
            pills += f'<span style="font-size:0.75rem;color:#484f58;margin-left:8px;">{signals_fired}/3 signals</span>'

        st.markdown(f"""
        <div class="conv-model {cls}">
          <div class="turn-label {label_cls}">🤖 Model Response — {status_icon}</div>
          <div class="conv-msg">{model_resp[:1200]}{"…" if len(model_resp) > 1200 else ""}</div>
          <div style="margin-top:8px;">{pills}</div>
        </div>""", unsafe_allow_html=True)


# ─── Tab 5: Cross-Run Comparison ──────────────────────────────────────────────

def tab_cross_run(runs):
    st.markdown('<div class="info-box"><p>📊 Compare results across multiple benchmark runs to track model safety improvements or regressions over time.</p></div>', unsafe_allow_html=True)

    all_metrics_files = {rid: paths["metrics"] for rid, paths in runs.items()}
    if len(all_metrics_files) < 2:
        st.info("Need at least 2 runs with metrics files for comparison. Run more experiments.")
        _render_single_run_summary(all_metrics_files)
        return

    sel_runs = st.multiselect(
        "Select runs to compare",
        list(all_metrics_files.keys()),
        default=list(all_metrics_files.keys())[:2],
        format_func=lambda x: f"Run {x}",
    )
    if not sel_runs:
        return

    all_rows = []
    for run_id in sel_runs:
        m = load_metrics(run_id, str(all_metrics_files[run_id]))
        for key, val in m.get("asr_by_model_defense", {}).items():
            all_rows.append({
                "Run": run_id[:12],
                "Model": label_model(val["model"]),
                "Defense": label_defense(val["defense"]),
                "ASR": val["asr"] * 100,
            })

    if not all_rows:
        st.info("No data in selected runs.")
        return
    df = pd.DataFrame(all_rows)

    fig = px.bar(
        df, x="Run", y="ASR", color="Defense", barmode="group",
        facet_col="Model", facet_col_wrap=3,
        labels={"ASR": "Attack Success Rate (%)"},
        color_discrete_sequence=[ACCENT_GREEN, ACCENT_BLUE, ACCENT_ORANGE],
    )
    fig.update_traces(marker_line_width=0)
    fig = apply_chart_style(fig, height=420)
    fig.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig, width='stretch')


def _render_single_run_summary(all_metrics_files):
    if not all_metrics_files:
        return
    run_id = list(all_metrics_files.keys())[0]
    m = load_metrics(run_id, str(all_metrics_files[run_id]))
    st.markdown(f"**Available run:** `{run_id}`")
    for key, val in m.get("asr_by_model_defense", {}).items():
        st.write(f"- {label_model(val['model'])} · {label_defense(val['defense'])}: ASR = {pct(val['asr'])}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    runs = discover_runs()
    metrics, filters = render_sidebar(runs)

    if metrics is None:
        st.error("No benchmark results found. Run `python run_experiments.py` to generate results.")
        return

    # Hero
    models_str = " · ".join(label_model(m) for m in metrics.get("models", []))
    n_attack = metrics.get("n_attack", 0)
    n_benign = metrics.get("n_benign", 0)

    st.markdown(f"""
    <div class="hero-banner">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <div class="hero-title">🛡️ SafeBench AI</div>
          <div class="hero-subtitle">Multi-turn Jailbreak Safety Benchmarking · Reproducible · Three-Signal Scoring</div>
          <div style="margin-top:16px;">
            <span class="hero-badge">Run {metrics.get('run_id','')[:16]}</span>
            <span class="hero-badge hero-badge-blue">{models_str or "No models"}</span>
            <span class="hero-badge hero-badge-green">{n_attack + n_benign} sequences</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.78rem;color:#484f58;line-height:2;">
            {n_attack} attack sequences · {n_benign} benign sequences<br>
            3 defense configurations · 3-signal voting (keyword · Detoxify · LLM judge)<br>
            Metrics: ASR · Detection Latency · Over-Refusal · Defense Ablation
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    filters_active = bool(filters["models"] and filters["defenses"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Overview  ",
        "  Attack Analysis  ",
        "  Defense Analysis  ",
        "  Conversation Explorer  ",
        "  Cross-Run Compare  ",
    ])

    with tab1:
        tab_overview(metrics, filters, filters_active)
    with tab2:
        if filters_active:
            tab_attack_analysis(metrics, filters)
        else:
            st.info("Select at least one model and one defense in the sidebar to see charts.")
    with tab3:
        if filters_active:
            tab_defense_analysis(metrics, filters)
        else:
            st.info("Select at least one model and one defense in the sidebar to see charts.")
    with tab4:
        if filters_active:
            tab_conversation_explorer(metrics, filters)
        else:
            st.info("Select at least one model and one defense in the sidebar to see charts.")
    with tab5:
        tab_cross_run(runs)


if __name__ == "__main__":
    main()

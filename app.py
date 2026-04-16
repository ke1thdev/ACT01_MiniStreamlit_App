import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph Simulator",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f0f1a; }

    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid #2a2a4a;
    }

    /* Cards */
    .sim-card {
        background: linear-gradient(135deg, #1e1e3a, #252545);
        border: 1px solid #3a3a6a;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    .sim-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .sim-subtitle {
        color: #7070a0;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }

    .formula-box {
        background: #0d0d1f;
        border: 1px solid #4040a0;
        border-radius: 8px;
        padding: 10px 16px;
        font-family: 'Courier New', monospace;
        color: #a78bfa;
        font-size: 1.05rem;
        margin: 8px 0 16px 0;
        text-align: center;
    }

    .stat-chip {
        display: inline-block;
        background: #2a2a5a;
        border: 1px solid #4040a0;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #a0a0d0;
        margin: 3px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #7070a0;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { margin-top: 4px; }

    /* Number inputs */
    .stNumberInput input {
        background: #1a1a2e !important;
        color: #e0e0ff !important;
        border: 1px solid #3a3a6a !important;
        border-radius: 8px !important;
    }

    /* Select */
    .stSelectbox select, .stSelectbox > div > div {
        background: #1a1a2e !important;
        color: #e0e0ff !important;
    }

    h1, h2, h3, h4 { color: #e0e0ff !important; }
    p, label, .stMarkdown { color: #b0b0d0 !important; }

    hr { border-color: #2a2a4a !important; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#12122a",
    "axes.facecolor":   "#1a1a36",
    "axes.edgecolor":   "#3a3a6a",
    "axes.labelcolor":  "#a0a0d0",
    "axes.titlecolor":  "#c0c0f0",
    "xtick.color":      "#7070a0",
    "ytick.color":      "#7070a0",
    "grid.color":       "#2a2a4a",
    "grid.linewidth":   0.8,
    "text.color":       "#c0c0f0",
    "legend.facecolor": "#12122a",
    "legend.edgecolor": "#3a3a6a",
})

ACCENT_COLORS = ["#a78bfa", "#60a5fa", "#34d399", "#f97316", "#f43f5e", "#facc15"]

def styled_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.4, linestyle="--")
    return fig, ax

def finalize(ax, xlabel="x", ylabel="y", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.7)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sim-title" style="font-size:1.3rem">📈 Graph Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">Interactive Mathematical Visualizations</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧩 Functions")
    funcs = [
        "1️⃣  Linear & Quadratic",
        "2️⃣  Trigonometric",
        "3️⃣  Exponential & Log",
        "4️⃣  Parametric Curves",
        "5️⃣  Normal Distribution",
        "6️⃣  Polynomial Roots",
    ]
    for f in funcs:
        st.markdown(f'<div class="stat-chip">{f}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with Streamlit · matplotlib · numpy")

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="sim-title" style="font-size:2.2rem; text-align:center">📈 Graph Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#7070a0; margin-bottom:24px">Explore · Compute · Visualize</p>', unsafe_allow_html=True)

tabs = st.tabs([
    "📏 Linear & Quadratic",
    "〰️ Trigonometric",
    "📐 Exp & Log",
    "🌀 Parametric",
    "🔔 Normal Dist.",
    "🔢 Polynomial",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LINEAR & QUADRATIC
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Linear & Quadratic Plotter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">y = ax² + bx + c &nbsp;|&nbsp; y = mx + b</div>', unsafe_allow_html=True)

    mode = st.radio("Function type", ["Linear  y = mx + b", "Quadratic  y = ax² + bx + c"],
                    horizontal=True)

    col1, col2, col3 = st.columns(3)
    if "Linear" in mode:
        with col1: m = st.slider("Slope (m)", -10.0, 10.0, 1.0, 0.1)
        with col2: b = st.slider("Intercept (b)", -20.0, 20.0, 0.0, 0.5)
        formula = f"y = {m}x + {b}"
    else:
        with col1: a = st.slider("a", -5.0, 5.0, 1.0, 0.1)
        with col2: b = st.slider("b", -10.0, 10.0, 0.0, 0.5)
        with col3: c = st.slider("c", -20.0, 20.0, 0.0, 0.5)
        formula = f"y = {a}x² + {b}x + {c}"

    x_range = st.slider("x range", -20.0, 20.0, (-10.0, 10.0))

    if st.button("Plot Graph", key="lq"):
        x = np.linspace(x_range[0], x_range[1], 500)
        fig, ax = styled_fig()
        if "Linear" in mode:
            y = m * x + b
            ax.plot(x, y, color=ACCENT_COLORS[0], lw=2.5, label=formula)
            ax.axhline(0, color="#444466", lw=0.8)
            ax.axvline(0, color="#444466", lw=0.8)
            x_int = -b / m if m != 0 else None
            if x_int is not None:
                ax.scatter([x_int], [0], color=ACCENT_COLORS[2], s=80, zorder=5,
                           label=f"x-intercept: ({x_int:.2f}, 0)")
            ax.scatter([0], [b], color=ACCENT_COLORS[3], s=80, zorder=5,
                       label=f"y-intercept: (0, {b:.2f})")
            st.markdown(f'<div class="formula-box">{formula} &nbsp;|&nbsp; Slope = {m} &nbsp;|&nbsp; y-int = {b}</div>',
                        unsafe_allow_html=True)
        else:
            y = a * x**2 + b * x + c
            ax.plot(x, y, color=ACCENT_COLORS[1], lw=2.5, label=formula)
            ax.axhline(0, color="#444466", lw=0.8)
            ax.axvline(0, color="#444466", lw=0.8)
            disc = b**2 - 4*a*c
            vertex_x = -b / (2*a) if a != 0 else 0
            vertex_y = a * vertex_x**2 + b * vertex_x + c
            ax.scatter([vertex_x], [vertex_y], color=ACCENT_COLORS[4], s=100, zorder=5,
                       label=f"Vertex ({vertex_x:.2f}, {vertex_y:.2f})")
            roots_text = ""
            if disc >= 0:
                r1 = (-b + np.sqrt(disc)) / (2*a)
                r2 = (-b - np.sqrt(disc)) / (2*a)
                ax.scatter([r1, r2], [0, 0], color=ACCENT_COLORS[2], s=80, zorder=5,
                           label=f"Roots: {r1:.2f}, {r2:.2f}")
                roots_text = f"Roots: {r1:.3f}, {r2:.3f}"
            else:
                roots_text = "No real roots (Δ < 0)"
            st.markdown(f'<div class="formula-box">{formula} &nbsp;|&nbsp; Discriminant = {disc:.3f} &nbsp;|&nbsp; {roots_text}</div>',
                        unsafe_allow_html=True)
        finalize(ax, title=formula)
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRIGONOMETRIC
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Trigonometric Functions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">y = A · f(Bx + C) + D</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        trig_fns = st.multiselect("Functions", ["sin", "cos", "tan"], default=["sin", "cos"])
        A = st.slider("Amplitude (A)", 0.1, 5.0, 1.0, 0.1)
        B = st.slider("Frequency (B)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        C = st.slider("Phase shift (C)", -np.pi, np.pi, 0.0, 0.1)
        D = st.slider("Vertical shift (D)", -5.0, 5.0, 0.0, 0.1)
        cycles = st.slider("Cycles shown", 1, 6, 2)

    if st.button("Plot Trig", key="trig"):
        x = np.linspace(0, cycles * 2 * np.pi, 1000)
        fig, ax = styled_fig()
        fn_map = {"sin": np.sin, "cos": np.cos, "tan": np.tan}
        for i, fn in enumerate(trig_fns):
            y = A * fn_map[fn](B * x + C) + D
            if fn == "tan":
                # mask discontinuities
                y = np.where(np.abs(y) > 10, np.nan, y)
            ax.plot(x, y, color=ACCENT_COLORS[i], lw=2.2,
                    label=f"y = {A}·{fn}({B}x + {C:.2f}) + {D}")
        ax.axhline(0, color="#444466", lw=0.8)
        xticks = np.arange(0, cycles * 2 * np.pi + 0.1, np.pi / 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v/np.pi:.1f}π" for v in xticks])
        ax.set_ylim(-A * 2 - abs(D) - 1, A * 2 + abs(D) + 1)
        st.markdown(f'<div class="formula-box">Period = {2*np.pi/B:.3f} &nbsp;|&nbsp; Amplitude = {A} &nbsp;|&nbsp; Phase = {C:.3f} rad</div>',
                    unsafe_allow_html=True)
        finalize(ax, xlabel="x (radians)", title="Trigonometric Functions")
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPONENTIAL & LOGARITHMIC
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Exponential & Logarithmic</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">y = a·bˣ &nbsp;|&nbsp; y = a·ln(bx) &nbsp;|&nbsp; y = a·log₁₀(bx)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        exp_type = st.selectbox("Function", ["Exponential  y = a·bˣ", "Natural Log  y = a·ln(bx)",
                                             "Log base-10  y = a·log₁₀(bx)", "All three"])
        ea = st.slider("Coefficient (a)", 0.1, 5.0, 1.0, 0.1)
        eb = st.slider("Base / scale (b)", 0.1, 5.0, 2.0, 0.1)
    with col2:
        ex_range = st.slider("x range", -10.0, 10.0, (0.1, 6.0))
        show_deriv = st.checkbox("Show derivative", False)

    if st.button("Plot", key="exp"):
        x = np.linspace(max(ex_range[0], 0.001), ex_range[1], 600)
        fig, ax = styled_fig()
        ci = 0
        if exp_type in ["Exponential  y = a·bˣ", "All three"]:
            y = ea * (eb ** x)
            ax.plot(x, y, color=ACCENT_COLORS[ci], lw=2.2, label=f"y = {ea}·{eb}ˣ")
            if show_deriv:
                dy = ea * (eb ** x) * np.log(eb)
                ax.plot(x, dy, color=ACCENT_COLORS[ci], lw=1.4, linestyle="--",
                        label=f"dy/dx = {ea}·ln({eb})·{eb}ˣ", alpha=0.7)
            ci += 1
        if exp_type in ["Natural Log  y = a·ln(bx)", "All three"]:
            xp = x[x > 0]
            y = ea * np.log(eb * xp)
            ax.plot(xp, y, color=ACCENT_COLORS[ci], lw=2.2, label=f"y = {ea}·ln({eb}x)")
            if show_deriv:
                dy = ea / xp
                ax.plot(xp, dy, color=ACCENT_COLORS[ci], lw=1.4, linestyle="--",
                        label=f"dy/dx = {ea}/x", alpha=0.7)
            ci += 1
        if exp_type in ["Log base-10  y = a·log₁₀(bx)", "All three"]:
            xp = x[x > 0]
            y = ea * np.log10(eb * xp)
            ax.plot(xp, y, color=ACCENT_COLORS[ci], lw=2.2, label=f"y = {ea}·log₁₀({eb}x)")
            ci += 1
        ax.axhline(0, color="#444466", lw=0.8)
        ax.axvline(0, color="#444466", lw=0.8)
        ax.set_ylim(-15, 30)
        finalize(ax, title="Exponential & Logarithmic Functions")
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PARAMETRIC CURVES
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Parametric Curves</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">x(t), y(t) — Lissajous, Rose, Spirals & more</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        curve_type = st.selectbox("Curve preset", [
            "Lissajous Figure",
            "Rose Curve",
            "Archimedean Spiral",
            "Hypotrochoid",
            "Epicycloid",
        ])
        color_mode = st.selectbox("Color", ["Gradient (t)", "Solid violet", "Solid cyan"])
    with col2:
        if curve_type == "Lissajous Figure":
            pa = st.slider("Freq x (a)", 1, 8, 3)
            pb = st.slider("Freq y (b)", 1, 8, 2)
            delta = st.slider("Phase (δ)", 0.0, np.pi, np.pi/4, 0.05)
        elif curve_type == "Rose Curve":
            pk = st.slider("Petals (k)", 1, 12, 5)
            pr = st.slider("Radius", 0.5, 5.0, 3.0, 0.1)
        elif curve_type == "Archimedean Spiral":
            p_a = st.slider("a (gap)", 0.1, 2.0, 0.5, 0.05)
            p_turns = st.slider("Turns", 1, 10, 5)
        elif curve_type == "Hypotrochoid":
            p_R = st.slider("R (outer)", 3.0, 10.0, 5.0, 0.5)
            p_r = st.slider("r (inner)", 0.5, 5.0, 3.0, 0.5)
            p_d = st.slider("d (dist)", 0.5, 5.0, 2.0, 0.5)
        elif curve_type == "Epicycloid":
            p_R2 = st.slider("R (fixed)", 2.0, 8.0, 4.0, 0.5)
            p_r2 = st.slider("r (rolling)", 0.5, 4.0, 1.0, 0.5)

    if st.button("Plot Curve", key="param"):
        fig, ax = styled_fig((8, 8))
        ax.set_aspect("equal")
        t = np.linspace(0, 2 * np.pi * 8, 5000)

        if curve_type == "Lissajous Figure":
            t2 = np.linspace(0, 2 * np.pi, 5000)
            x = np.sin(pa * t2 + delta)
            y = np.sin(pb * t2)
            label = f"Lissajous a={pa}, b={pb}, δ={delta:.2f}"
        elif curve_type == "Rose Curve":
            theta = np.linspace(0, 2 * np.pi, 5000)
            r = pr * np.cos(pk * theta)
            x, y = r * np.cos(theta), r * np.sin(theta)
            t2 = theta
            label = f"Rose k={pk}"
        elif curve_type == "Archimedean Spiral":
            theta = np.linspace(0, p_turns * 2 * np.pi, 5000)
            r = p_a * theta
            x, y = r * np.cos(theta), r * np.sin(theta)
            t2 = theta
            label = f"Spiral a={p_a}, {p_turns} turns"
        elif curve_type == "Hypotrochoid":
            t2 = np.linspace(0, 2 * np.pi * int(p_R / np.gcd(int(p_R), int(p_r)) if p_r > 0 else 10), 5000)
            x = (p_R - p_r) * np.cos(t2) + p_d * np.cos((p_R - p_r) / p_r * t2)
            y = (p_R - p_r) * np.sin(t2) - p_d * np.sin((p_R - p_r) / p_r * t2)
            label = f"Hypotrochoid R={p_R}, r={p_r}, d={p_d}"
        elif curve_type == "Epicycloid":
            t2 = np.linspace(0, 2 * np.pi * int(p_R2 / p_r2) + 1, 5000)
            x = (p_R2 + p_r2) * np.cos(t2) - p_r2 * np.cos((p_R2 + p_r2) / p_r2 * t2)
            y = (p_R2 + p_r2) * np.sin(t2) - p_r2 * np.sin((p_R2 + p_r2) / p_r2 * t2)
            label = f"Epicycloid R={p_R2}, r={p_r2}"

        if color_mode == "Gradient (t)":
            from matplotlib.collections import LineCollection
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap="cool", linewidth=1.8, alpha=0.9)
            lc.set_array(np.linspace(0, 1, len(x)))
            ax.add_collection(lc)
            ax.autoscale()
        else:
            clr = ACCENT_COLORS[0] if "violet" in color_mode else ACCENT_COLORS[1]
            ax.plot(x, y, color=clr, lw=1.8, alpha=0.9, label=label)

        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — NORMAL DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Normal Distribution Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">Visualize μ, σ, and probability areas</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        mu = st.number_input("Mean (μ)", value=0.0, step=0.5)
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01, step=0.1)
    with col2:
        n_curves = st.slider("Number of curves", 1, 4, 1)
        shade_area = st.checkbox("Shade area (a to b)", True)
    with col3:
        lo = st.number_input("Lower bound (a)", value=-1.0, step=0.1)
        hi = st.number_input("Upper bound (b)", value=1.0, step=0.1)

    extra_mus    = [mu]
    extra_sigmas = [sigma]
    if n_curves > 1:
        for i in range(1, n_curves):
            c1, c2 = st.columns(2)
            with c1: extra_mus.append(st.number_input(f"μ {i+1}", value=float(i*2), step=0.5, key=f"mu{i}"))
            with c2: extra_sigmas.append(st.number_input(f"σ {i+1}", value=1.0 + i*0.3, min_value=0.01,
                                                          step=0.1, key=f"sig{i}"))

    if st.button("Simulate", key="norm"):
        fig, ax = styled_fig((10, 5))
        x_all = np.linspace(min(extra_mus) - 4 * max(extra_sigmas),
                            max(extra_mus) + 4 * max(extra_sigmas), 1000)
        for i, (m_, s_) in enumerate(zip(extra_mus, extra_sigmas)):
            y = norm.pdf(x_all, m_, s_)
            ax.plot(x_all, y, color=ACCENT_COLORS[i], lw=2.5, label=f"μ={m_}, σ={s_}")
            if shade_area and i == 0:
                x_shade = np.linspace(lo, hi, 300)
                y_shade = norm.pdf(x_shade, m_, s_)
                ax.fill_between(x_shade, y_shade, alpha=0.35, color=ACCENT_COLORS[i])
                prob = norm.cdf(hi, m_, s_) - norm.cdf(lo, m_, s_)
                ax.annotate(f"P({lo}≤X≤{hi}) = {prob:.4f}", xy=((lo+hi)/2, max(y_shade)/2),
                            ha="center", fontsize=10, color="#e0e0ff",
                            bbox=dict(boxstyle="round,pad=0.4", fc="#2a2a5a", ec="#5050a0", alpha=0.9))
            # mark σ bands
            for k, alpha_ in [(1, 0.15), (2, 0.08), (3, 0.04)]:
                ax.axvspan(m_ - k*s_, m_ + k*s_, alpha=alpha_, color=ACCENT_COLORS[i])

        if shade_area:
            prob = norm.cdf(hi, extra_mus[0], extra_sigmas[0]) - norm.cdf(lo, extra_mus[0], extra_sigmas[0])
            st.markdown(f'<div class="formula-box">P({lo} ≤ X ≤ {hi}) = {prob:.6f} &nbsp;|&nbsp; '
                        f'Z-scores: [{(lo-extra_mus[0])/extra_sigmas[0]:.3f}, {(hi-extra_mus[0])/extra_sigmas[0]:.3f}]</div>',
                        unsafe_allow_html=True)
        finalize(ax, xlabel="x", ylabel="Probability Density", title="Normal Distribution")
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — POLYNOMIAL ROOTS
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.markdown('<div class="sim-title">Polynomial Plotter & Root Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sim-subtitle">Enter coefficients for any degree polynomial</div>', unsafe_allow_html=True)

    degree = st.slider("Degree", 1, 8, 4)

    st.markdown("**Coefficients** (highest degree → constant)")
    cols = st.columns(degree + 1)
    coeffs = []
    for i, col in enumerate(cols):
        exp = degree - i
        label = f"x^{exp}" if exp > 1 else ("x" if exp == 1 else "const")
        val = col.number_input(label, value=float(1 if i == 0 else 0), step=0.5,
                               key=f"coeff_{i}", format="%.2f")
        coeffs.append(val)

    pr_range = st.slider("x range for plot", -20.0, 20.0, (-5.0, 5.0))

    if st.button("Find Roots & Plot", key="poly"):
        p = np.poly1d(coeffs)
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        complex_roots = roots[np.abs(roots.imag) >= 1e-9]

        x = np.linspace(pr_range[0], pr_range[1], 800)
        y = p(x)
        y_clipped = np.clip(y, -1e4, 1e4)

        fig, ax = styled_fig()
        ax.plot(x, y_clipped, color=ACCENT_COLORS[0], lw=2.5, label=str(p))
        ax.axhline(0, color="#555580", lw=1.2, linestyle="--")
        ax.axvline(0, color="#555580", lw=1.2, linestyle="--")

        for rx in real_roots:
            if pr_range[0] <= rx <= pr_range[1]:
                ax.scatter([rx], [0], color=ACCENT_COLORS[2], s=100, zorder=6,
                           label=f"Root: x = {rx:.4f}")
                ax.annotate(f"  {rx:.3f}", (rx, 0), color=ACCENT_COLORS[2],
                            fontsize=8, va="bottom")

        # Critical points
        dp = np.polyder(p)
        crit = np.roots(dp.coeffs)
        real_crit = crit[np.abs(crit.imag) < 1e-9].real
        for cx in real_crit:
            if pr_range[0] <= cx <= pr_range[1]:
                cy = p(cx)
                ax.scatter([cx], [cy], color=ACCENT_COLORS[3], s=80, zorder=5,
                           marker="^", label=f"Critical: ({cx:.3f}, {cy:.3f})")

        ax.set_ylim(np.percentile(y_clipped, 2) - 1, np.percentile(y_clipped, 98) + 1)

        poly_str = str(p).replace("\n", " ")
        st.markdown(f'<div class="formula-box">{poly_str}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Real Roots**")
            if len(real_roots) > 0:
                for r in real_roots:
                    st.markdown(f'<div class="stat-chip">x = {r:.6f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="stat-chip">No real roots</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Complex Roots**")
            if len(complex_roots) > 0:
                for r in complex_roots:
                    st.markdown(f'<div class="stat-chip">{r.real:.4f} ± {abs(r.imag):.4f}i</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="stat-chip">None</div>', unsafe_allow_html=True)

        finalize(ax, title=f"Degree-{degree} Polynomial")
        st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
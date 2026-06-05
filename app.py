"""
Medical Impact — Service Quality Intelligence Platform
SERVQUAL survey of community health brigades | Streamlit + scikit-learn (NO train_test_split)

Dataset: medical_impact.xlsx (427 patient survey responses across 8 communities, 3 states).
Target (regression): satisfaccion_general — fit on the FULL dataset (in-sample only).
"""

import datetime as _dt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Impacto Médico · Calidad del Servicio",
                   page_icon="✚", layout="wide", initial_sidebar_state="expanded")

# ---- columns ----
FACTORS = ['F1_Tangibilidad', 'F2_Fiabilidad', 'F3_CapRespuesta', 'F4_Seguridad', 'F5_Empatia']
FACTOR_LABEL = {'F1_Tangibilidad': 'Tangibilidad', 'F2_Fiabilidad': 'Fiabilidad',
                'F3_CapRespuesta': 'Cap. Respuesta', 'F4_Seguridad': 'Seguridad',
                'F5_Empatia': 'Empatía'}
CAT_FEATURES = ['region', 'escolaridad', 'genero', 'Estado_Comunidad']
TARGET = 'satisfaccion_general'

# ---- design tokens ----
INK="#0B1220"; SLATE="#475569"; MUTED="#94A3B8"; LINE="#E9E2E4"
SURFACE="#FFFFFF"; CANVAS="#FBF7F8"; NAVY="#8B1A1A"; ACCENT="#D7263D"
TEAL="#B91C1C"; AMBER="#B45309"; GRID="#F2EAEC"

mpl.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "axes.edgecolor": LINE, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "600", "axes.titlecolor": INK,
    "axes.labelsize": 10, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})
def style_ax(ax):
    ax.tick_params(length=0); ax.grid(axis="x", visible=False)
    return ax

# ---- data + model (cached) ----
@st.cache_data(show_spinner=False)
def load_local(path='medical_impact.xlsx'):
    return pd.read_excel(path)

@st.cache_resource(show_spinner=False)
def train_model(df):
    """Predict overall satisfaction from the 5 SERVQUAL factors + demographics.
    Fit on the FULL dataset — no train_test_split (in-sample metrics only)."""
    d = df.dropna(subset=[TARGET]).copy()
    d[FACTORS] = d[FACTORS].apply(lambda s: s.fillna(s.mean()))
    X, y = d[FACTORS + CAT_FEATURES], d[TARGET]
    pre = ColumnTransformer(transformers=[
        ('num', StandardScaler(), FACTORS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_FEATURES)
    ])
    model = Pipeline(steps=[('preprocess', pre), ('model', LinearRegression())])
    model.fit(X, y)
    yp = model.predict(X)
    metrics = {'r2': r2_score(y, yp), 'mae': mean_absolute_error(y, yp),
               'rmse': mean_squared_error(y, yp) ** 0.5}
    return model, metrics

def s(x): return f"{x:.2f}"

# ---- CSS design system (mirrors the retail platform) ----
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --muted:{MUTED}; --line:{LINE};
  --surface:{SURFACE}; --canvas:{CANVAS}; --navy:{NAVY}; --accent:{ACCENT};
  --teal:{TEAL}; --amber:{AMBER}; --radius:18px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
  --shadow-lg:0 2px 6px rgba(16,24,40,.06), 0 18px 48px rgba(16,24,40,.10); }}
.stApp {{ background:
   radial-gradient(1200px 600px at 80% -10%, #FCE9EC 0%, rgba(252,233,236,0) 55%), var(--canvas); }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1320px; }}
html, body, [class*="css"] {{ font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;
   color:var(--ink); -webkit-font-smoothing:antialiased; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden; height:0; }}
[data-testid="stToolbar"] {{ display:none; }}
::selection {{ background:{ACCENT}; color:#fff; }}
::-moz-selection {{ background:{ACCENT}; color:#fff; }}

.hero {{ position:relative; overflow:hidden; border-radius:24px; margin-bottom:22px;
  padding:30px 34px; color:#EAFBF5;
  background: radial-gradient(900px 300px at 90% -40%, rgba(215,38,61,.40), rgba(215,38,61,0) 60%),
    linear-gradient(135deg, #0B1220 0%, #5A0F12 55%, #8B1A1A 120%);
  box-shadow:var(--shadow-lg); }}
.hero .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:#F5B5BE; font-weight:700; margin-bottom:8px; }}
.hero h1 {{ font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; line-height:1.1; margin:0 0 8px; color:#fff; letter-spacing:-.01em; }}
.hero p {{ font-size:14.5px; color:#F3C9CF; margin:0; max-width:700px; line-height:1.5; }}
.hero .brand {{ position:absolute; top:22px; right:26px; display:flex; align-items:center; gap:10px; font-weight:700; color:#F6D7DC; font-size:13px; }}
.hero .brand .dot {{ width:30px; height:30px; border-radius:9px; background:linear-gradient(135deg,#D7263D,#E5566B); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; }}
.hero .meta {{ position:absolute; bottom:22px; right:26px; font-size:12px; color:#F5B5BE; display:flex; align-items:center; gap:8px; }}
.hero .live {{ width:8px; height:8px; border-radius:50%; background:#E5566B; box-shadow:0 0 0 4px rgba(215,38,61,.18); }}

.sec {{ display:flex; align-items:center; gap:12px; margin:26px 2px 12px; }}
.sec .bar {{ width:4px; height:18px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); }}
.sec h2 {{ font-family:'Manrope',sans-serif; font-size:15px; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:var(--ink); margin:0; }}
.sec .hint {{ font-size:12.5px; color:var(--muted); margin-left:auto; }}

.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }}
@media (max-width:1100px) {{ .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{ position:relative; background:rgba(255,255,255,.78); backdrop-filter:blur(10px);
  border:1px solid var(--line); border-radius:var(--radius); padding:18px;
  box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s, border-color .18s; }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:var(--shadow-lg); border-color:#CDE8E0; }}
.kpi .top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi .label {{ font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--slate); }}
.kpi .ico {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:16px; background:#FCE9EC; color:var(--navy); }}
.kpi .val {{ font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; color:var(--ink); margin:12px 0 4px; }}
.kpi .sub {{ font-size:12px; color:var(--muted); display:flex; gap:6px; align-items:center; }}
.kpi .chip {{ font-weight:700; padding:1px 7px; border-radius:999px; font-size:11px; }}
.chip-pos {{ color:#0F766E; background:#D7F2EE; }} .chip-warn {{ color:#92400E; background:#FBECCB; }} .chip-neu {{ color:#334155; background:#E9EEF6; }}
.kpi .accent {{ position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); opacity:0; transition:opacity .2s; }}
.kpi:hover .accent {{ opacity:1; }}

.ins-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
@media (max-width:1100px) {{ .ins-grid {{ grid-template-columns:1fr; }} }}
.ins {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }}
.ins:hover {{ transform:translateY(-3px); box-shadow:var(--shadow-lg); }}
.ins .k {{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; display:flex; gap:8px; align-items:center; }}
.ins .t {{ font-size:15px; font-weight:700; color:var(--ink); margin:0 0 6px; }}
.ins .d {{ font-size:13px; color:var(--slate); line-height:1.5; margin:0; }}

.card {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:18px 20px 6px; height:100%; }}
.card .h {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }}
.card .h .ttl {{ font-size:14px; font-weight:700; color:var(--ink); }}
.card .h .tag {{ font-size:11px; font-weight:600; color:var(--muted); border:1px solid var(--line); padding:2px 8px; border-radius:999px; }}
.note {{ font-size:12.5px; color:var(--slate); line-height:1.5; border-top:1px dashed var(--line); margin-top:4px; padding:10px 2px 4px; }}
.note b {{ color:var(--ink); }}

.fc-wrap {{ background:linear-gradient(135deg,#0B1220 0%, #5A0F12 100%); border-radius:22px; padding:6px; box-shadow:var(--shadow-lg); }}
.fc-value {{ background:linear-gradient(180deg,#0E1A38,#0B1220); border-radius:18px; padding:26px 24px; color:#fff; height:100%; display:flex; flex-direction:column; justify-content:center; }}
.fc-value .lab {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:#F5B5BE; font-weight:700; }}
.fc-value .big {{ font-family:'Manrope',sans-serif; font-size:44px; font-weight:800; line-height:1; margin:10px 0 12px; color:#fff; }}
.fc-value .pill {{ display:inline-flex; gap:8px; align-items:center; font-size:12.5px; font-weight:600; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#F6D7DC; width:fit-content; }}

.limit {{ display:flex; gap:12px; align-items:flex-start; background:#FFF8EC; border:1px solid #F3E2BD; border-left:4px solid var(--amber); border-radius:14px; padding:14px 16px; color:#7C5310; font-size:13px; line-height:1.5; }}
.limit .i {{ font-size:16px; }}

section[data-testid="stSidebar"] {{ background:#0B1220; border-right:1px solid #3A1417; }}
section[data-testid="stSidebar"] * {{ color:#F6D7DC; }}
section[data-testid="stSidebar"] .side-brand {{ display:flex; gap:10px; align-items:center; padding:6px 4px 14px; border-bottom:1px solid #3A1417; margin-bottom:12px; }}
section[data-testid="stSidebar"] .side-brand .dot {{ width:30px;height:30px;border-radius:9px; background:linear-gradient(135deg,#D7263D,#E5566B); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; }}
section[data-testid="stSidebar"] .side-brand .n {{ font-weight:800; color:#fff; font-size:14px; }}
section[data-testid="stSidebar"] .mini {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#C98A92; font-weight:700; margin:14px 2px 6px; }}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{ background:#2A1012; border:1px solid #3A1417; border-radius:12px; padding:10px 12px; }}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color:#fff; font-size:18px; }}
.stMultiSelect [data-baseweb="tag"] {{ background:#8B1A1A !important; }}

.stButton>button {{ background:linear-gradient(135deg,var(--accent),var(--navy)); color:#fff; border:0; border-radius:12px; padding:.6rem 1.1rem; font-weight:700; box-shadow:0 8px 20px rgba(215,38,61,.28); transition:transform .15s, box-shadow .15s; }}
.stButton>button:hover {{ transform:translateY(-2px); box-shadow:0 12px 26px rgba(215,38,61,.36); }}
[data-testid="stExpander"] {{ border:1px solid var(--line) !important; border-radius:16px !important; box-shadow:var(--shadow); background:var(--surface); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color:var(--ink) !important; font-weight:700 !important; }}
[data-testid="stExpander"] summary svg {{ fill:var(--slate) !important; }}
.stNumberInput input {{ color:#fff !important; background:#2A1012 !important; border-radius:10px !important; }}
.stSelectbox div[data-baseweb="select"] *, .stSelectbox div[data-baseweb="select"] {{ color:#fff !important; }}
.stNumberInput label, .stSelectbox label, .stMultiSelect label {{ color:var(--ink) !important; }}
</style>
""", unsafe_allow_html=True)

# ---- sidebar: data + filters ----
st.sidebar.markdown("<div class='side-brand'><div class='dot'>✚</div>"
    "<div><div class='n'>Impacto Médico</div>"
    "<div style='font-size:11px;color:#C98A92'>Calidad del Servicio</div></div></div>", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Fuente de datos · medical_impact.xlsx", type=['xlsx'])
if uploaded is not None:
    df = pd.read_excel(uploaded)
else:
    try:
        df = load_local()
    except FileNotFoundError:
        st.markdown("<div class='hero'><div class='eyebrow'>Impacto Médico</div>"
            "<h1>Inteligencia de Calidad del Servicio</h1><p>Sube <b>medical_impact.xlsx</b> "
            "en la barra lateral para comenzar, o colócalo junto a app.py.</p></div>", unsafe_allow_html=True)
        st.stop()

model, metrics = train_model(df)

st.sidebar.markdown("<div class='mini'>Filtros</div>", unsafe_allow_html=True)
states = st.sidebar.multiselect("Estado", sorted(df.Estado_Comunidad.dropna().unique()),
                                sorted(df.Estado_Comunidad.dropna().unique()))
comms  = st.sidebar.multiselect("Comunidad", sorted(df.comunidad.dropna().unique()),
                                sorted(df.comunidad.dropna().unique()))
regs   = st.sidebar.multiselect("Región", sorted(df.region.dropna().unique()),
                                sorted(df.region.dropna().unique()))

f = df[df.Estado_Comunidad.isin(states) & df.comunidad.isin(comms) & df.region.isin(regs)]

st.sidebar.markdown("<div class='mini'>Modelo · en muestra (sin división)</div>", unsafe_allow_html=True)
sm1, sm2, sm3 = st.sidebar.columns(3)
sm1.metric("R²", f"{metrics['r2']:.3f}")
sm2.metric("MAE", s(metrics['mae']))
sm3.metric("RMSE", s(metrics['rmse']))

# ---- hero ----
_now = _dt.datetime.now().strftime("%b %d, %Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="brand"><span class="dot">✚</span> Impacto Médico</div>
  <div class="eyebrow">Salud · Reporte de Calidad SERVQUAL</div>
  <h1>Brigadas de Salud Comunitaria — Inteligencia de Calidad del Servicio</h1>
  <p>Calidad del servicio reportada por los pacientes en las brigadas de salud comunitaria,
     medida con las cinco dimensiones SERVQUAL. Diseñado para responder, de un vistazo, cómo
     califican el servicio los pacientes, qué comunidades destacan y qué impulsa la satisfacción general.</p>
  <div class="meta"><span class="live"></span> Actualizado {_now} · {len(df)} respuestas</div>
</div>
""", unsafe_allow_html=True)

if f.empty:
    st.markdown("<div class='limit'><div class='i'>⚠</div><div>Ninguna respuesta coincide con los "
                "filtros seleccionados. Amplía los filtros en la barra lateral.</div></div>", unsafe_allow_html=True)
    st.stop()

# ---- KPIs ----
servqual = f['SERVQUAL AVG'].mean()
satis = f[TARGET].mean()
recommend = f.recomendacion.isin(['Sí, seguro', 'Creo que sí']).mean() * 100
n_comm = f.comunidad.nunique()
n_resp = len(f)

def chip(cls, txt): return f"<span class='chip {cls}'>{txt}</span>"
sq_chip = chip("chip-pos","Excelente") if servqual >= 4.3 else (chip("chip-neu","Buena") if servqual>=3.8 else chip("chip-warn","Baja"))
sat_chip = chip("chip-pos","Alta") if satis >= 4.3 else chip("chip-neu","Media")
rec_chip = chip("chip-pos","Fuerte") if recommend >= 90 else chip("chip-warn","Atención")

st.markdown("<div class='sec'><span class='bar'></span><h2>KPIs de Calidad</h2>"
            f"<span class='hint'>{n_resp} de {len(df)} respuestas en vista</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Índice SERVQUAL</span><span class="ico">◎</span></div>
    <div class="val">{servqual:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div>
    <div class="sub">{sq_chip} calidad general del servicio</div></div>
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Satisfacción General</span><span class="ico">★</span></div>
    <div class="val">{satis:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div>
    <div class="sub">{sat_chip} reportada por pacientes</div></div>
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Recomendaría</span><span class="ico">♥</span></div>
    <div class="val">{recommend:.1f}%</div>
    <div class="sub">{rec_chip} "sí" + "creo que sí"</div></div>
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Comunidades</span><span class="ico">⌂</span></div>
    <div class="val">{n_comm}</div>
    <div class="sub">{chip('chip-neu','Cobertura')} en vista actual</div></div>
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Respuestas</span><span class="ico">▤</span></div>
    <div class="val">{n_resp}</div>
    <div class="sub">{chip('chip-neu','Muestra')} registros de encuesta</div></div>
</div>
""", unsafe_allow_html=True)

# ---- insights ----
dim_means = f[FACTORS].mean().rename(index=FACTOR_LABEL)
best_dim, worst_dim = dim_means.idxmax(), dim_means.idxmin()
comm_q = f.groupby('comunidad')['SERVQUAL AVG'].mean().sort_values(ascending=False)
st.markdown("<div class='sec'><span class='bar'></span><h2>Hallazgos Clave</h2>"
            "<span class='hint'>generado automáticamente · descriptivo, no causal</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="ins-grid">
  <div class="ins"><div class="k">✚ Dimensión más fuerte</div>
    <div class="t">{best_dim} lidera la calidad del servicio</div>
    <div class="d">Dimensión SERVQUAL mejor calificada con {dim_means.max():.2f}/5 en la selección actual.</div></div>
  <div class="ins"><div class="k">✚ Área de mejora</div>
    <div class="t">{worst_dim} es la dimensión más baja</div>
    <div class="d">Peor calificada con {dim_means.min():.2f}/5 — la oportunidad más clara para elevar la calidad general.</div></div>
  <div class="ins"><div class="k">✚ Comunidad líder</div>
    <div class="t">{comm_q.index[0]} tiene la mejor calificación</div>
    <div class="d">Lidera con {comm_q.iloc[0]:.2f}/5; la más baja es {comm_q.index[-1]} con {comm_q.iloc[-1]:.2f}/5.</div></div>
</div>
""", unsafe_allow_html=True)

# ---- analytics ----
st.markdown("<div class='sec'><span class='bar'></span><h2>Analítica</h2>"
            "<span class='hint'>dimensiones · comunidades · demografía</span></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

# Chart 1 — SERVQUAL by dimension
with col1:
    st.markdown("<div class='card'><div class='h'><span class='ttl'>Puntaje Promedio por Dimensión SERVQUAL</span>"
                "<span class='tag'>1–5</span></div>", unsafe_allow_html=True)
    dm = f[FACTORS].mean().rename(index=FACTOR_LABEL).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.2)); style_ax(ax)
    bars = ax.bar(dm.index, dm.values, color=NAVY, width=.66, zorder=3); bars[0].set_color(ACCENT)
    ax.set_ylim(0, 5); ax.set_ylabel("Puntaje medio"); plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown(f"<div class='note'><b>{dm.idxmax()}</b> es la dimensión mejor calificada ({dm.max():.2f}/5); "
                f"<b>{dm.idxmin()}</b> la más baja ({dm.min():.2f}/5). Solo descriptivo.</div></div>", unsafe_allow_html=True)

# Chart 2 — SERVQUAL by community
with col2:
    st.markdown("<div class='card'><div class='h'><span class='ttl'>Calidad del Servicio por Comunidad</span>"
                "<span class='tag'>SERVQUAL AVG</span></div>", unsafe_allow_html=True)
    cq = f.groupby('comunidad')['SERVQUAL AVG'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.4, 3.2)); style_ax(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.barh(cq.index, cq.values, color=SLATE, zorder=3)
    ax.patches[-1].set_color(TEAL)
    ax.set_xlim(0, 5); ax.set_xlabel("SERVQUAL medio"); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown(f"<div class='note'><b>{cq.index[-1]}</b> tiene la calificación más alta ({cq.iloc[-1]:.2f}); "
                f"<b>{cq.index[0]}</b> la más baja ({cq.iloc[0]:.2f}). Promedios por comunidad.</div></div>", unsafe_allow_html=True)

# Chart 3 — Satisfaction by state
st.markdown("<div class='card'><div class='h'><span class='ttl'>Satisfacción General por Estado</span>"
            "<span class='tag'>mean</span></div>", unsafe_allow_html=True)
stt = f.groupby('Estado_Comunidad')[TARGET].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 3.2)); style_ax(ax)
bars = ax.bar(stt.index, stt.values, color=NAVY, width=.5, zorder=3); bars[0].set_color(ACCENT)
ax.set_ylim(0, 5); ax.set_ylabel("Satisfacción media")
for i, v in enumerate(stt.values): ax.text(i, v + .07, f"{v:.2f}", ha='center', fontsize=9, color=INK)
plt.tight_layout(); st.pyplot(fig, use_container_width=True)
st.markdown(f"<div class='note'><b>{stt.index[0]}</b> reporta la satisfacción media más alta ({stt.iloc[0]:.2f}/5) "
            f"and <b>{stt.index[-1]}</b> la más baja ({stt.iloc[-1]:.2f}/5) en el filtro actual.</div></div>",
            unsafe_allow_html=True)

# Extra analytics
with st.expander("Más analítica · mapa de calor de correlaciones"):
    st.markdown(f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px'>"
                "Mapa de Calor · dimensiones SERVQUAL y satisfacción</div>",
                unsafe_allow_html=True)
    num_cols = FACTORS + ['SERVQUAL AVG', 'satisfaccion_general']
    corr = f[num_cols].corr()
    labels = [FACTOR_LABEL.get(c, c.replace('satisfaccion_general','Satisfacción').replace('SERVQUAL AVG','SERVQUAL')) for c in num_cols]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    im = ax.imshow(corr, cmap='Reds', vmin=0, vmax=1)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8,
                    color='white' if v > 0.6 else '#0B1220')
    ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)
    st.markdown("<div class='note'>Valores más altos = dimensiones que se mueven juntas. Las dimensiones más "
                "alineadas con la satisfacción general son las palancas más fuertes para mejorarla (asociación, no prueba).</div>",
                unsafe_allow_html=True)

# ---- forecast / prediction center ----
st.markdown("<div class='sec'><span class='bar'></span><h2>Predictor de Satisfacción · Centro de Escenarios</h2>"
            "<span class='hint'>estima la satisfacción a partir de las dimensiones de calidad</span></div>", unsafe_allow_html=True)
left, right = st.columns([1.45, 1], gap="large")
with left:
    st.markdown("<div class='card' style='padding-bottom:18px'><div class='h'>"
                "<span class='ttl'>Entradas del escenario · dimensiones SERVQUAL (1–5)</span>"
                "<span class='tag'>9 entradas</span></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.slider("Tangibilidad", 1.0, 5.0, 4.5, 0.25)
        f2 = st.slider("Fiabilidad", 1.0, 5.0, 4.5, 0.25)
        f3 = st.slider("Cap. de Respuesta", 1.0, 5.0, 4.4, 0.25)
        f4 = st.slider("Seguridad", 1.0, 5.0, 4.6, 0.25)
        f5 = st.slider("Empatía", 1.0, 5.0, 4.4, 0.25)
    with c2:
        reg = st.selectbox("Región", sorted(df.region.dropna().unique()),
                           index=sorted(df.region.dropna().unique()).index("Centro") if "Centro" in df.region.values else 0)
        esc = st.selectbox("Escolaridad", sorted(df.escolaridad.dropna().unique()))
        gen = st.selectbox("Género", sorted(df.genero.dropna().unique()))
        est = st.selectbox("Estado", sorted(df.Estado_Comunidad.dropna().unique()),
                           index=sorted(df.Estado_Comunidad.dropna().unique()).index("Puebla") if "Puebla" in df.Estado_Comunidad.values else 0)
    go = st.button("✚  Predecir Satisfacción", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

scenario = pd.DataFrame([{
    'F1_Tangibilidad': f1, 'F2_Fiabilidad': f2, 'F3_CapRespuesta': f3,
    'F4_Seguridad': f4, 'F5_Empatia': f5,
    'region': reg, 'escolaridad': esc, 'genero': gen, 'Estado_Comunidad': est
}])
pred = float(model.predict(scenario[FACTORS + CAT_FEATURES])[0])
pred = max(1.0, min(5.0, pred))  # display-clamp to the 1–5 scale
level = "Alta" if pred >= 4.3 else ("Moderada" if pred >= 3.5 else "Baja")
lc = "#16A34A" if level == "Alta" else ("#F59E0B" if level == "Moderada" else "#DC2626")
with right:
    st.markdown(f"""
    <div class="fc-wrap"><div class="fc-value">
      <div class="lab">Satisfacción Prevista</div>
      <div class="big">{pred:.2f}<span style="font-size:20px;color:#F5B5BE"> / 5</span></div>
      <div class="pill"><span style="width:8px;height:8px;border-radius:50%;background:{lc};display:inline-block"></span>
        satisfacción esperada {level}</div>
      <div style="margin-top:14px;font-size:12.5px;color:#F5B5BE;line-height:1.5">
        Modelo en muestra · R² {metrics['r2']:.3f} · error típico ≈ {s(metrics['mae'])} puntos.
        Estimación orientativa, no una garantía validada.</div>
    </div></div>
    """, unsafe_allow_html=True)
st.markdown(f"<div class='note' style='border:0;padding-top:10px'>Este perfil implica "
            f"una satisfacción esperada <b>{level.lower()}</b> (<b>{pred:.2f}/5</b>). Las dimensiones con "
            f"mayor peso en el modelo mueven más esta estimación — priorízalas para elevar la satisfacción.</div>",
            unsafe_allow_html=True)

# ---- model summary + limitation ----
st.markdown("<div class='sec'><span class='bar'></span><h2>Resumen del Modelo</h2></div>", unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3, gap="large")
for c, lab, val in [(mc1, "R² en muestra", f"{metrics['r2']:.3f}"),
                    (mc2, "MAE en muestra", s(metrics['mae'])),
                    (mc3, "RMSE en muestra", s(metrics['rmse']))]:
    c.markdown(f"<div class='kpi'><span class='accent'></span>"
               f"<div class='top'><span class='label'>{lab}</span><span class='ico'>∑</span></div>"
               f"<div class='val'>{val}</div><div class='sub'>ajuste con todos los datos</div></div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{SLATE};font-size:13.5px;line-height:1.6;margin:16px 2px 12px'>"
    f"El modelo predice la <b style='color:{INK}'>satisfacción general</b> a partir de las cinco dimensiones SERVQUAL "
    f"más región, escolaridad, género y estado. Explica alrededor del "
    f"<b style='color:{INK}'>{metrics['r2']*100:.0f}%</b> de la variación en la satisfacción, con un "
    f"error típico de <b style='color:{INK}'>{s(metrics['mae'])}</b> puntos en la escala de 1–5. Las calificaciones son muy "
    f"altas y se concentran cerca del máximo, lo que limita naturalmente cuánta variación puede explicar cualquier modelo.</p>",
    unsafe_allow_html=True)
st.markdown(
    "<div class='limit'><div class='i'>⚠</div><div><b>Limitación.</b> Estas métricas de regresión son "
    "<b>en muestra</b> — el modelo se entrenó con todos los datos <b>sin división train/test</b>. Describen "
    "el ajuste a esta encuesta y no deben tomarse como precisión futura validada. Los puntajes son autoreportados y "
    "están sesgados al alza, por lo que los resultados son orientativos, no causales.</div></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:12px;margin-top:26px;"
    f"padding-top:16px;border-top:1px solid {LINE}'>"
    "Impacto Médico · Estudio de Calidad SERVQUAL &nbsp;·&nbsp; métricas en muestra</div>",
    unsafe_allow_html=True)

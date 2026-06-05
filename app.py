"""
Impacto Medico - Plataforma de Inteligencia de Calidad del Servicio (Suite Pro)
Encuesta SERVQUAL de brigadas de salud comunitaria | Streamlit + scikit-learn (SIN train_test_split)

UI en espanol; codigo/identificadores en ingles.
Dataset: medical_impact.xlsx (427 respuestas, 8 comunidades, 3 estados, 19 items de servicio).
Objetivo de regresion: satisfaccion_general, ajustado sobre TODO el dataset (metricas en muestra).
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
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Impacto Medico - Calidad del Servicio",
                   page_icon="✚", layout="wide", initial_sidebar_state="expanded")

# ---- columns (data keys: keep as-is) ----
FACTORS = ['F1_Tangibilidad', 'F2_Fiabilidad', 'F3_CapRespuesta', 'F4_Seguridad', 'F5_Empatia']
FACTOR_LABEL = {'F1_Tangibilidad':'Tangibilidad','F2_Fiabilidad':'Fiabilidad',
                'F3_CapRespuesta':'Cap. Respuesta','F4_Seguridad':'Seguridad','F5_Empatia':'Empatía'}
CAT_FEATURES = ['region', 'escolaridad', 'genero', 'Estado_Comunidad']
TARGET = 'satisfaccion_general'
ITEMS = ['material_limpio','lugar_adecuado','personal_presentable','materiales_suficientes','problema_resuelto',
         'puntualidad','cumplieron_promesas','registro_correcto','atencion_rapida','personal_dispuesto',
         'dudas_resueltas','trato_amable','confianza_personal','seguridad_atencion','conocimiento_personal',
         'atencion_personalizada','entendieron_necesidades','preocupacion_paciente','horario_conveniente']
ITEM_LABEL = {
 'material_limpio':'Material limpio','lugar_adecuado':'Lugar adecuado','personal_presentable':'Personal presentable',
 'materiales_suficientes':'Materiales suficientes','problema_resuelto':'Problema resuelto','puntualidad':'Puntualidad',
 'cumplieron_promesas':'Cumplieron promesas','registro_correcto':'Registro correcto','atencion_rapida':'Atención rápida',
 'personal_dispuesto':'Personal dispuesto','dudas_resueltas':'Dudas resueltas','trato_amable':'Trato amable',
 'confianza_personal':'Confianza en el personal','seguridad_atencion':'Seguridad en la atención',
 'conocimiento_personal':'Conocimiento del personal','atencion_personalizada':'Atención personalizada',
 'entendieron_necesidades':'Entendieron necesidades','preocupacion_paciente':'Preocupación por el paciente',
 'horario_conveniente':'Horario conveniente'}
GM_ORDER = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']

# ---- red / white tokens ----
INK="#0B1220"; SLATE="#475569"; MUTED="#94A3B8"; LINE="#E9E2E4"
SURFACE="#FFFFFF"; CANVAS="#FBF7F8"; NAVY="#8B1A1A"; ACCENT="#D7263D"; AMBER="#B45309"; GRID="#F2EAEC"

mpl.rcParams.update({
    "figure.facecolor":"none","axes.facecolor":"none","savefig.facecolor":"none",
    "axes.edgecolor":LINE,"axes.linewidth":1.0,"axes.grid":True,"grid.color":GRID,"grid.linewidth":1.0,
    "axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.titleweight":"600",
    "axes.titlecolor":INK,"axes.labelsize":10,"axes.labelcolor":SLATE,"xtick.color":SLATE,"ytick.color":SLATE,
    "xtick.labelsize":9,"ytick.labelsize":9,"font.family":"sans-serif",
    "font.sans-serif":["Inter","Segoe UI","Helvetica Neue","Arial","DejaVu Sans"]})
def style_ax(ax): ax.tick_params(length=0); ax.grid(axis="x", visible=False); return ax
def fig_close(fig): st.pyplot(fig, use_container_width=True); plt.close(fig)
def s(x): return f"{x:.2f}"

@st.cache_data(show_spinner=False)
def load_local(path='medical_impact.xlsx'): return pd.read_excel(path)

@st.cache_resource(show_spinner=False)
def train_model(df):
    d = df.dropna(subset=[TARGET]).copy()
    d[FACTORS] = d[FACTORS].apply(lambda s: s.fillna(s.mean()))
    X, y = d[FACTORS + CAT_FEATURES], d[TARGET]
    pre = ColumnTransformer([('num', StandardScaler(), FACTORS),
                             ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_FEATURES)])
    model = Pipeline([('preprocess', pre), ('model', LinearRegression())]); model.fit(X, y)
    yp = model.predict(X)
    return model, {'r2': r2_score(y, yp), 'mae': mean_absolute_error(y, yp),
                   'rmse': mean_squared_error(y, yp) ** 0.5}, d, yp

@st.cache_resource(show_spinner=False)
def driver_importance(_model, d):
    X, y = d[FACTORS + CAT_FEATURES], d[TARGET]
    imp = permutation_importance(_model, X, y, n_repeats=10, random_state=42, scoring='r2')
    return (pd.DataFrame({'Feature': X.columns, 'imp': imp.importances_mean})
            .sort_values('imp', ascending=False).reset_index(drop=True))

# ---- CSS (identical identity to prior app) ----
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --muted:{MUTED}; --line:{LINE}; --surface:{SURFACE}; --canvas:{CANVAS};
  --navy:{NAVY}; --accent:{ACCENT}; --radius:18px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
  --shadow-lg:0 2px 6px rgba(16,24,40,.06), 0 18px 48px rgba(16,24,40,.10); }}
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #FCE9EC 0%, rgba(252,233,236,0) 55%), var(--canvas); }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1340px; }}
html, body, [class*="css"] {{ font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); -webkit-font-smoothing:antialiased; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden; height:0; }}
[data-testid="stToolbar"] {{ display:none; }}
::selection {{ background:{ACCENT}; color:#fff; }} ::-moz-selection {{ background:{ACCENT}; color:#fff; }}
.hero {{ position:relative; overflow:hidden; border-radius:24px; margin-bottom:22px; padding:30px 34px; color:#FCE9EC;
  background: radial-gradient(900px 300px at 90% -40%, rgba(215,38,61,.40), rgba(215,38,61,0) 60%),
    linear-gradient(135deg,#0B1220 0%,#5A0F12 55%,#8B1A1A 120%); box-shadow:var(--shadow-lg); }}
.hero .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:#F5B5BE; font-weight:700; margin-bottom:8px; }}
.hero h1 {{ font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; line-height:1.1; margin:0 0 8px; color:#fff; letter-spacing:-.01em; }}
.hero p {{ font-size:14.5px; color:#F3C9CF; margin:0; max-width:740px; line-height:1.5; }}
.hero .brand {{ position:absolute; top:22px; right:26px; display:flex; align-items:center; gap:10px; font-weight:700; color:#F6D7DC; font-size:13px; }}
.hero .brand .dot {{ width:30px;height:30px;border-radius:9px; background:linear-gradient(135deg,#D7263D,#E5566B); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; }}
.hero .meta {{ position:absolute; bottom:22px; right:26px; font-size:12px; color:#F5B5BE; display:flex; align-items:center; gap:8px; }}
.hero .live {{ width:8px; height:8px; border-radius:50%; background:#E5566B; box-shadow:0 0 0 4px rgba(215,38,61,.18); }}
.sec {{ display:flex; align-items:center; gap:12px; margin:22px 2px 12px; }}
.sec .bar {{ width:4px; height:18px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); }}
.sec h2 {{ font-family:'Manrope',sans-serif; font-size:15px; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:var(--ink); margin:0; }}
.sec .hint {{ font-size:12.5px; color:var(--muted); margin-left:auto; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }}
@media (max-width:1100px) {{ .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{ position:relative; background:rgba(255,255,255,.80); backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s, border-color .18s; }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:var(--shadow-lg); border-color:#EBC9D0; }}
.kpi .top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi .label {{ font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--slate); }}
.kpi .ico {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:16px; background:#FCE9EC; color:var(--navy); }}
.kpi .val {{ font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; color:var(--ink); margin:12px 0 4px; }}
.kpi .sub {{ font-size:12px; color:var(--muted); display:flex; gap:6px; align-items:center; }}
.kpi .chip {{ font-weight:700; padding:1px 7px; border-radius:999px; font-size:11px; }}
.chip-pos {{ color:#9F1239; background:#FCE0E6; }} .chip-warn {{ color:#92400E; background:#FBECCB; }} .chip-neu {{ color:#334155; background:#E9EEF6; }}
.kpi .accent {{ position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); opacity:0; transition:opacity .2s; }}
.kpi:hover .accent {{ opacity:1; }}
.ins-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
@media (max-width:1100px) {{ .ins-grid {{ grid-template-columns:1fr; }} }}
.ins {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }}
.ins:hover {{ transform:translateY(-3px); box-shadow:var(--shadow-lg); }}
.ins .k {{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }}
.ins .t {{ font-size:15px; font-weight:700; color:var(--ink); margin:0 0 6px; }}
.ins .d {{ font-size:13px; color:var(--slate); line-height:1.5; margin:0; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:16px 18px 8px; height:100%; }}
.ctitle {{ font-size:14px; font-weight:700; color:var(--ink); margin-bottom:6px; }}
.note {{ font-size:12.5px; color:var(--slate); line-height:1.5; border-top:1px dashed var(--line); margin-top:2px; padding:10px 2px 2px; }}
.note b {{ color:var(--ink); }}
.fc-wrap {{ background:linear-gradient(135deg,#0B1220 0%, #5A0F12 100%); border-radius:22px; padding:6px; box-shadow:var(--shadow-lg); }}
.fc-value {{ background:linear-gradient(180deg,#2A1012,#0B1220); border-radius:18px; padding:26px 24px; color:#fff; height:100%; display:flex; flex-direction:column; justify-content:center; }}
.fc-value .lab {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:#F5B5BE; font-weight:700; }}
.fc-value .big {{ font-family:'Manrope',sans-serif; font-size:44px; font-weight:800; line-height:1; margin:10px 0 12px; color:#fff; }}
.fc-value .pill {{ display:inline-flex; gap:8px; align-items:center; font-size:12.5px; font-weight:600; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#F6D7DC; width:fit-content; }}
.limit {{ display:flex; gap:12px; align-items:flex-start; background:#FFF8EC; border:1px solid #F3E2BD; border-left:4px solid var(--amber); border-radius:14px; padding:14px 16px; color:#7C5310; font-size:13px; line-height:1.5; }}
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
.stNumberInput input {{ color:#fff !important; background:#2A1012 !important; border-radius:10px !important; }}
.stSelectbox div[data-baseweb="select"] *, .stSelectbox div[data-baseweb="select"] {{ color:#fff !important; }}
.stNumberInput label, .stSelectbox label, .stMultiSelect label, .stSlider label {{ color:var(--ink) !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--line); flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:var(--slate); }}
.stTabs [aria-selected="true"] {{ color:var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent); }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }}
</style>
""", unsafe_allow_html=True)

# ---- sidebar ----
st.sidebar.markdown("<div class='side-brand'><div class='dot'>✚</div>"
    "<div><div class='n'>Impacto Médico</div><div style='font-size:11px;color:#C98A92'>Calidad del Servicio</div></div></div>",
    unsafe_allow_html=True)
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

model, metrics, d_model, y_pred_full = train_model(df)

st.sidebar.markdown("<div class='mini'>Filtros</div>", unsafe_allow_html=True)
states = st.sidebar.multiselect("Estado", sorted(df.Estado_Comunidad.dropna().unique()), sorted(df.Estado_Comunidad.dropna().unique()))
comms  = st.sidebar.multiselect("Comunidad", sorted(df.comunidad.dropna().unique()), sorted(df.comunidad.dropna().unique()))
regs   = st.sidebar.multiselect("Región", sorted(df.region.dropna().unique()), sorted(df.region.dropna().unique()))
f = df[df.Estado_Comunidad.isin(states) & df.comunidad.isin(comms) & df.region.isin(regs)]

st.sidebar.markdown("<div class='mini'>Modelo · en muestra (sin división)</div>", unsafe_allow_html=True)
sm1, sm2, sm3 = st.sidebar.columns(3)
sm1.metric("R²", f"{metrics['r2']:.3f}"); sm2.metric("MAE", s(metrics['mae'])); sm3.metric("RMSE", s(metrics['rmse']))

# ---- hero ----
_now = _dt.datetime.now().strftime("%d/%m/%Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="brand"><span class="dot">✚</span> Impacto Médico</div>
  <div class="eyebrow">Salud · Reporte de Calidad SERVQUAL · Suite Analítica</div>
  <h1>Brigadas de Salud Comunitaria — Inteligencia de Calidad del Servicio</h1>
  <p>Análisis integral de la calidad del servicio reportada por pacientes en brigadas de salud comunitaria:
     dimensiones SERVQUAL, los 19 reactivos del cuestionario, comunidades y estados, contexto de marginación,
     y un modelo de pronóstico de satisfacción. Diseñado para responder qué pasó, por qué, y dónde actuar.</p>
  <div class="meta"><span class="live"></span> Actualizado {_now} · {len(df)} respuestas</div>
</div>
""", unsafe_allow_html=True)

if f.empty:
    st.markdown("<div class='limit'><div>⚠ Ninguna respuesta coincide con los filtros. Amplía los filtros en la barra lateral.</div></div>", unsafe_allow_html=True)
    st.stop()

# ---- KPIs ----
servqual = f['SERVQUAL AVG'].mean(); satis = f[TARGET].mean()
recommend = f.recomendacion.isin(['Sí, seguro','Creo que sí']).mean()*100
top_box = (f.recomendacion=='Sí, seguro').mean()*100
n_comm = f.comunidad.nunique(); n_resp = len(f)
def chip(c,t): return f"<span class='chip {c}'>{t}</span>"
sq_chip = chip("chip-pos","Excelente") if servqual>=4.3 else (chip("chip-neu","Buena") if servqual>=3.8 else chip("chip-warn","Baja"))
sat_chip = chip("chip-pos","Alta") if satis>=4.3 else chip("chip-neu","Media")
rec_chip = chip("chip-pos","Fuerte") if recommend>=90 else chip("chip-warn","Atención")
st.markdown("<div class='sec'><span class='bar'></span><h2>KPIs de Calidad</h2>"
            f"<span class='hint'>{n_resp} de {len(df)} respuestas en vista</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Índice SERVQUAL</span><span class="ico">◎</span></div>
    <div class="val">{servqual:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div><div class="sub">{sq_chip} calidad general</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Satisfacción</span><span class="ico">★</span></div>
    <div class="val">{satis:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div><div class="sub">{sat_chip} reportada por pacientes</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Recomendaría</span><span class="ico">♥</span></div>
    <div class="val">{recommend:.1f}%</div><div class="sub">{rec_chip} · {top_box:.0f}% "sí, seguro"</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Comunidades</span><span class="ico">⌂</span></div>
    <div class="val">{n_comm}</div><div class="sub">{chip('chip-neu','Cobertura')} en vista actual</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Respuestas</span><span class="ico">▤</span></div>
    <div class="val">{n_resp}</div><div class="sub">{chip('chip-neu','Muestra')} registros</div></div>
</div>
""", unsafe_allow_html=True)

# ---- insights (dynamic) ----
dim_means = f[FACTORS].mean().rename(index=FACTOR_LABEL)
item_means = f[ITEMS].mean().rename(index=ITEM_LABEL)
comm_q = f.groupby('comunidad')['SERVQUAL AVG'].mean().sort_values(ascending=False)
st.markdown("<div class='sec'><span class='bar'></span><h2>Hallazgos Clave</h2>"
            "<span class='hint'>generado automáticamente · descriptivo, no causal</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="ins-grid">
  <div class="ins"><div class="k">✚ Fortaleza</div><div class="t">{dim_means.idxmax()} lidera la calidad</div>
    <div class="d">Dimensión mejor calificada con {dim_means.max():.2f}/5. A nivel reactivo, lo más alto es
      «{item_means.idxmax()}» ({item_means.max():.2f}).</div></div>
  <div class="ins"><div class="k">✚ Prioridad de mejora</div><div class="t">{item_means.idxmin()} es lo más bajo</div>
    <div class="d">El reactivo peor evaluado ({item_means.min():.2f}/5). La dimensión más baja es
      {dim_means.idxmin()} ({dim_means.min():.2f}).</div></div>
  <div class="ins"><div class="k">✚ Comunidad líder</div><div class="t">{comm_q.index[0]} destaca</div>
    <div class="d">Lidera con {comm_q.iloc[0]:.2f}/5; la más baja es {comm_q.index[-1]} con {comm_q.iloc[-1]:.2f}/5.</div></div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
st.markdown("<div class='sec'><span class='bar'></span><h2>Analítica</h2>"
            "<span class='hint'>dimensiones · reactivos · comunidades · relaciones · distribuciones · tendencia · modelo</span></div>",
            unsafe_allow_html=True)
T1,T2,T3,T4,T5,T6 = st.tabs(["📊 Dimensiones","📋 19 Reactivos","🗺️ Comunidades & Estados",
                             "🔗 Relaciones","📈 Distribuciones","🤖 Modelo & Pronóstico"])

# ---------- TAB 1: DIMENSIONS ----------
with T1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Perfil SERVQUAL (radar)</div>", unsafe_allow_html=True)
        dm = f[FACTORS].mean().rename(index=FACTOR_LABEL)
        labels=list(dm.index); vals=list(dm.values)
        ang=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist(); vc=vals+vals[:1]; ac=ang+ang[:1]
        fig,ax=plt.subplots(figsize=(5.2,4.2),subplot_kw=dict(polar=True))
        ax.plot(ac,vc,color=ACCENT,lw=2.2); ax.fill(ac,vc,color=ACCENT,alpha=.18)
        ax.set_xticks(ang); ax.set_xticklabels(labels,fontsize=9); ax.set_ylim(3.5,5)
        ax.set_yticks([4,4.5,5]); ax.set_yticklabels(["4.0","4.5","5.0"],fontsize=8,color=MUTED)
        ax.grid(color=GRID); ax.spines['polar'].set_color(LINE); fig_close(fig)
        st.markdown(f"<div class='note'>Punto alto: <b>{dm.idxmax()}</b> ({dm.max():.2f}); más bajo: "
                    f"<b>{dm.idxmin()}</b> ({dm.min():.2f}).</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Puntaje promedio por dimensión</div>", unsafe_allow_html=True)
        dms=f[FACTORS].mean().rename(index=FACTOR_LABEL).sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(5.2,4.2)); style_ax(ax)
        bars=ax.bar(dms.index,dms.values,color=NAVY,width=.66,zorder=3); bars[0].set_color(ACCENT)
        ax.set_ylim(0,5); ax.set_ylabel("Puntaje medio")
        for i,v in enumerate(dms.values): ax.text(i,v+.07,f"{v:.2f}",ha='center',fontsize=8,color=INK)
        plt.xticks(rotation=20,ha='right'); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{dms.idxmax()}</b> es la más alta; <b>{dms.idxmin()}</b> la más baja. Solo descriptivo.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Mapa de calor · Comunidad × Dimensión</div>", unsafe_allow_html=True)
    piv=f.groupby('comunidad')[FACTORS].mean().rename(columns=FACTOR_LABEL)
    piv=piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig,ax=plt.subplots(figsize=(11,4.4)); im=ax.imshow(piv.values,cmap='Reds',vmin=3.5,vmax=5,aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]; ax.text(j,i,f"{v:.2f}",ha='center',va='center',fontsize=8,color='white' if v>4.5 else INK)
    ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02); plt.tight_layout(); fig_close(fig)
    weak=piv.stack().idxmin()
    st.markdown(f"<div class='note'>Punto más bajo: <b>{weak[0]} · {weak[1]}</b> ({piv.stack().min():.2f}). Filas ordenadas por calidad media.</div></div>", unsafe_allow_html=True)

# ---------- TAB 2: 19 ITEMS ----------
with T2:
    st.markdown("<div class='card'><div class='ctitle'>Ranking de los 19 reactivos del cuestionario (puntaje medio)</div>", unsafe_allow_html=True)
    im_s=f[ITEMS].mean().rename(index=ITEM_LABEL).sort_values()
    fig,ax=plt.subplots(figsize=(11,6.2)); style_ax(ax); ax.grid(axis='y',visible=False); ax.grid(axis='x',visible=True)
    colors=[ACCENT if i<5 else (NAVY if i>=len(im_s)-5 else SLATE) for i in range(len(im_s))]
    ax.barh(im_s.index, im_s.values, color=colors, zorder=3)
    for i,v in enumerate(im_s.values): ax.text(v+0.01,i,f"{v:.2f}",va='center',fontsize=8,color=INK)
    ax.set_xlim(4.0,4.75); ax.set_xlabel("Puntaje medio (1–5)"); plt.tight_layout(); fig_close(fig)
    st.markdown(f"<div class='note'>Las 5 barras rojas inferiores son las <b>prioridades de mejora</b> "
                f"(la más baja: «{im_s.index[0]}», {im_s.values[0]:.2f}); las 5 superiores en granate son las fortalezas. "
                f"Convierte el SERVQUAL en acciones concretas.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Mapa de calor · Reactivo × Estado</div>", unsafe_allow_html=True)
    pis=f.groupby('Estado_Comunidad')[ITEMS].mean().T.rename(index=ITEM_LABEL)
    pis=pis.loc[pis.mean(axis=1).sort_values().index]
    fig,ax=plt.subplots(figsize=(9, 7)); im=ax.imshow(pis.values,cmap='Reds',vmin=3.8,vmax=5,aspect='auto')
    ax.set_xticks(range(len(pis.columns))); ax.set_xticklabels(pis.columns,fontsize=9)
    ax.set_yticks(range(len(pis.index))); ax.set_yticklabels(pis.index,fontsize=8)
    for i in range(pis.shape[0]):
        for j in range(pis.shape[1]):
            v=pis.values[i,j]; ax.text(j,i,f"{v:.1f}",ha='center',va='center',fontsize=7,color='white' if v>4.55 else INK)
    ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>Dónde cada estado es fuerte o débil por reactivo. Reactivos ordenados de menor a mayor "
                "promedio (los de arriba requieren más atención).</div></div>", unsafe_allow_html=True)

# ---------- TAB 3: COMMUNITIES & STATES ----------
with T3:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>SERVQUAL por comunidad</div>", unsafe_allow_html=True)
        cq=f.groupby('comunidad')['SERVQUAL AVG'].mean().sort_values()
        fig,ax=plt.subplots(figsize=(5.4,3.8)); style_ax(ax); ax.grid(axis='y',visible=False); ax.grid(axis='x',visible=True)
        ax.barh(cq.index,cq.values,color=SLATE,zorder=3); ax.patches[-1].set_color(ACCENT)
        ax.set_xlim(0,5); ax.set_xlabel("SERVQUAL medio")
        for i,v in enumerate(cq.values): ax.text(v+0.03,i,f"{v:.2f}",va='center',fontsize=8,color=INK)
        plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{cq.index[-1]}</b> lidera ({cq.iloc[-1]:.2f}); <b>{cq.index[0]}</b> la más baja ({cq.iloc[0]:.2f}).</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Satisfacción por estado</div>", unsafe_allow_html=True)
        stt=f.groupby('Estado_Comunidad')[TARGET].mean().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(5.4,3.8)); style_ax(ax)
        bars=ax.bar(stt.index,stt.values,color=NAVY,width=.5,zorder=3); bars[0].set_color(ACCENT)
        ax.set_ylim(0,5); ax.set_ylabel("Satisfacción media")
        for i,v in enumerate(stt.values): ax.text(i,v+.07,f"{v:.2f}",ha='center',fontsize=9,color=INK)
        plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{stt.index[0]}</b> la más alta ({stt.iloc[0]:.2f}); <b>{stt.index[-1]}</b> la más baja ({stt.iloc[-1]:.2f}).</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Calidad por grado de marginación (GM_2020)</div>", unsafe_allow_html=True)
    gm=f.groupby('GM_2020')['SERVQUAL AVG'].mean().reindex([g for g in GM_ORDER if g in f.GM_2020.unique()]).dropna()
    gmn=f.groupby('GM_2020').size().reindex(gm.index)
    fig,ax=plt.subplots(figsize=(11,3.6)); style_ax(ax)
    bars=ax.bar(gm.index,gm.values,color=ACCENT,width=.6,zorder=3)
    ax.set_ylim(0,5); ax.set_ylabel("SERVQUAL medio"); ax.set_xlabel("Grado de marginación (menor → mayor)")
    for i,(v,nn) in enumerate(zip(gm.values,gmn.values)): ax.text(i,v+.07,f"{v:.2f}\n(n={int(nn)})",ha='center',fontsize=8,color=INK)
    plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>La calidad percibida se mantiene alta en todos los grados de marginación, incluso "
                "«Muy alto». Hallazgo pivotal del impacto social de las brigadas (descriptivo, no causal).</div></div>", unsafe_allow_html=True)

# ---------- TAB 4: RELATIONSHIPS ----------
with T4:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Mapa de calor de correlaciones</div>", unsafe_allow_html=True)
        num_cols=FACTORS+['SERVQUAL AVG',TARGET]
        corr=f[num_cols].corr()
        labs=[FACTOR_LABEL.get(c,c.replace(TARGET,'Satisfacción').replace('SERVQUAL AVG','SERVQUAL')) for c in num_cols]
        fig,ax=plt.subplots(figsize=(5.6,4.6)); im=ax.imshow(corr,cmap='Reds',vmin=0,vmax=1)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                v=corr.iloc[i,j]; ax.text(j,i,f"{v:.2f}",ha='center',va='center',fontsize=7.5,color='white' if v>0.6 else INK)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=35,ha='right',fontsize=8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=8); ax.grid(False)
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); plt.tight_layout(); fig_close(fig)
        st.markdown("<div class='note'>Dimensiones más alineadas con la satisfacción = palancas más fuertes (asociación, no prueba).</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Dispersión · SERVQUAL vs Satisfacción</div>", unsafe_allow_html=True)
        x=f['SERVQUAL AVG'].values; y=f[TARGET].values; m=~(np.isnan(x)|np.isnan(y)); x,y=x[m],y[m]
        fig,ax=plt.subplots(figsize=(5.6,4.6)); style_ax(ax)
        ax.scatter(x,y,s=22,alpha=.45,color=ACCENT,edgecolor='white',linewidth=.4,zorder=3)
        if len(x)>2:
            b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),50); ax.plot(xs,a+b*xs,color=NAVY,lw=2,zorder=4)
        r=np.corrcoef(x,y)[0,1] if len(x)>2 else float('nan')
        ax.set_xlabel("SERVQUAL AVG"); ax.set_ylabel("Satisfacción"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Correlación <b>r = {r:.2f}</b>: mayor calidad percibida acompaña mayor satisfacción (lineal, no causal).</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Marginación (IM_2020) vs Calidad — nivel comunidad</div>", unsafe_allow_html=True)
    cl=f.groupby('comunidad').agg(IM=('IM_2020','mean'),SERV=('SERVQUAL AVG','mean'),n=('id','count')).dropna()
    fig,ax=plt.subplots(figsize=(11,4.0)); style_ax(ax)
    ax.scatter(cl.IM,cl.SERV,s=cl.n*2.5,alpha=.55,color=ACCENT,edgecolor=NAVY,linewidth=1,zorder=3)
    for name,row in cl.iterrows(): ax.annotate(name,(row.IM,row.SERV),fontsize=8,color=SLATE,xytext=(4,4),textcoords='offset points')
    if len(cl)>2:
        b,a=np.polyfit(cl.IM,cl.SERV,1); xs=np.linspace(cl.IM.min(),cl.IM.max(),50); ax.plot(xs,a+b*xs,color=NAVY,lw=2,ls='--',zorder=2)
    rr=cl.IM.corr(cl.SERV)
    ax.set_xlabel("Índice de marginación IM_2020 (mayor = más marginada)"); ax.set_ylabel("SERVQUAL medio")
    plt.tight_layout(); fig_close(fig)
    st.markdown(f"<div class='note'>Correlación a nivel comunidad <b>r = {rr:.2f}</b>; tamaño del punto = nº de respuestas. "
                f"Las brigadas mantienen calidad alta incluso en zonas marginadas (asociación, no causa).</div></div>", unsafe_allow_html=True)

# ---------- TAB 5: DISTRIBUTIONS ----------
with T5:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Histograma · Índice SERVQUAL</div>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax); ax.grid(axis='x',visible=False)
        ax.hist(f['SERVQUAL AVG'].dropna(),bins=14,color=ACCENT,alpha=.85,edgecolor='white')
        ax.axvline(f['SERVQUAL AVG'].mean(),color=NAVY,lw=2,ls='--'); ax.set_xlabel("SERVQUAL AVG"); ax.set_ylabel("Frecuencia")
        plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Media {f['SERVQUAL AVG'].mean():.2f}. Concentrada en valores altos.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Caja y bigotes · Satisfacción por estado</div>", unsafe_allow_html=True)
        order=f.groupby('Estado_Comunidad')[TARGET].median().sort_values(ascending=False).index
        data=[f.loc[f.Estado_Comunidad==stx,TARGET].dropna().values for stx in order]
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax)
        bp=ax.boxplot(data,tick_labels=list(order),patch_artist=True,medianprops=dict(color=NAVY,linewidth=2))
        for p in bp['boxes']: p.set(facecolor='#FCE0E6',edgecolor=ACCENT)
        ax.set_ylabel("Satisfacción"); ax.set_ylim(0.8,5.2); plt.xticks(rotation=10); plt.tight_layout(); fig_close(fig)
        st.markdown("<div class='note'>Mediana y dispersión por estado. Cajas altas y compactas = satisfacción consistente.</div></div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("<div class='card'><div class='ctitle'>Satisfacción por grupo de edad</div>", unsafe_allow_html=True)
        age_order=['18 a 29 años','30 a 44 años','45 a 59 años','60 años o más']
        ag=f[f.edad.isin(age_order)].groupby('edad')[TARGET].mean().reindex(age_order).dropna()
        agn=f[f.edad.isin(age_order)].groupby('edad').size().reindex(ag.index)
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax)
        ax.plot(range(len(ag)),ag.values,marker='o',color=ACCENT,lw=2.2,zorder=3)
        ax.set_xticks(range(len(ag))); ax.set_xticklabels([a.replace(' años','') for a in ag.index],fontsize=8)
        ax.set_ylim(4.0,5.0); ax.set_ylabel("Satisfacción media")
        for i,(v,nn) in enumerate(zip(ag.values,agn.values)): ax.text(i,v+.02,f"{v:.2f}",ha='center',fontsize=8,color=INK)
        plt.tight_layout(); fig_close(fig)
        st.markdown("<div class='note'>Satisfacción por edad; los adultos jóvenes-medios tienden a calificar ligeramente más alto.</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><div class='ctitle'>Intención de recomendar</div>", unsafe_allow_html=True)
        rec_order=['Sí, seguro','Creo que sí','No sé','Creo que no','No, para nada','No especificado']
        rc=f.recomendacion.value_counts().reindex(rec_order).dropna()
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax); ax.grid(axis='y',visible=False); ax.grid(axis='x',visible=True)
        cols=[ACCENT if k in ('Sí, seguro','Creo que sí') else SLATE for k in rc.index]
        ax.barh(rc.index[::-1],rc.values[::-1],color=cols[::-1],zorder=3); ax.set_xlabel("Respuestas")
        plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{recommend:.1f}%</b> respondería \"sí\" o \"creo que sí\".</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Composición de la muestra (demografía)</div>", unsafe_allow_html=True)
    demo=[('genero','Género'),('escolaridad','Escolaridad'),('edad','Edad'),('primera vez','¿Primera vez?')]
    fig,axes=plt.subplots(1,4,figsize=(12,3.0))
    for ax,(col,title) in zip(axes,demo):
        vc=f[col].value_counts().sort_values()
        ax.barh(range(len(vc)),vc.values,color=ACCENT,zorder=3)
        ax.set_yticks(range(len(vc))); ax.set_yticklabels([str(x)[:16] for x in vc.index],fontsize=7.5)
        ax.set_title(title,fontsize=10,color=INK); ax.grid(axis='y',visible=False); ax.tick_params(length=0); ax.set_xlabel("n",fontsize=8)
    plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>Perfil de quienes respondieron — contexto para interpretar los resultados.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Tendencia por orden de recolección (id) · media móvil y acumulada</div>", unsafe_allow_html=True)
    seq=f.sort_values('id')[['id','SERVQUAL AVG']].dropna().reset_index(drop=True)
    win=max(5,len(seq)//20); roll=seq['SERVQUAL AVG'].rolling(win,min_periods=1).mean(); cum=seq['SERVQUAL AVG'].expanding().mean()
    fig,ax=plt.subplots(figsize=(11,3.6)); style_ax(ax)
    ax.scatter(seq.index,seq['SERVQUAL AVG'],s=10,alpha=.18,color=SLATE,zorder=2)
    ax.plot(seq.index,roll,color=ACCENT,lw=2.2,zorder=4,label=f"Media móvil ({win})")
    ax.plot(seq.index,cum,color=NAVY,lw=2,ls='--',zorder=3,label="Media acumulada")
    ax.set_xlabel("Orden de la respuesta (secuencia de id)"); ax.set_ylabel("SERVQUAL AVG"); ax.set_ylim(3.3,5.05)
    ax.legend(frameon=False,fontsize=9,loc='lower right'); plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>El dataset no tiene fecha; usamos el <b>orden de recolección (id)</b> como secuencia. "
                "La media acumulada se estabiliza: muestra consistente, sin deriva marcada.</div></div>", unsafe_allow_html=True)

# ---------- TAB 6: MODEL & FORECAST ----------
with T6:
    st.markdown("<div class='sec' style='margin-top:6px'><span class='bar'></span><h2>Predictor de Satisfacción · Centro de Escenarios</h2>"
                "<span class='hint'>estima la satisfacción a partir de las dimensiones</span></div>", unsafe_allow_html=True)
    left,right=st.columns([1.45,1],gap="large")
    with left:
        st.markdown("<div class='card' style='padding-bottom:18px'><div class='ctitle'>Entradas del escenario · dimensiones SERVQUAL (1–5)</div>", unsafe_allow_html=True)
        cc1,cc2=st.columns(2)
        with cc1:
            f1=st.slider("Tangibilidad",1.0,5.0,4.5,0.25); f2=st.slider("Fiabilidad",1.0,5.0,4.5,0.25)
            f3=st.slider("Cap. de Respuesta",1.0,5.0,4.4,0.25); f4=st.slider("Seguridad",1.0,5.0,4.6,0.25)
            f5=st.slider("Empatía",1.0,5.0,4.4,0.25)
        with cc2:
            reg=st.selectbox("Región",sorted(df.region.dropna().unique()),
                index=sorted(df.region.dropna().unique()).index("Centro") if "Centro" in df.region.values else 0)
            esc=st.selectbox("Escolaridad",sorted(df.escolaridad.dropna().unique()))
            gen=st.selectbox("Género",sorted(df.genero.dropna().unique()))
            est=st.selectbox("Estado",sorted(df.Estado_Comunidad.dropna().unique()),
                index=sorted(df.Estado_Comunidad.dropna().unique()).index("Puebla") if "Puebla" in df.Estado_Comunidad.values else 0)
        st.button("✚  Predecir Satisfacción",type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    scenario=pd.DataFrame([{'F1_Tangibilidad':f1,'F2_Fiabilidad':f2,'F3_CapRespuesta':f3,'F4_Seguridad':f4,'F5_Empatia':f5,
        'region':reg,'escolaridad':esc,'genero':gen,'Estado_Comunidad':est}])
    pred=max(1.0,min(5.0,float(model.predict(scenario[FACTORS+CAT_FEATURES])[0])))
    level="Alta" if pred>=4.3 else ("Moderada" if pred>=3.5 else "Baja")
    lc="#16A34A" if level=="Alta" else ("#F59E0B" if level=="Moderada" else "#DC2626")
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
        </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec' style='margin-top:18px'><span class='bar'></span><h2>Diagnóstico del Modelo</h2></div>", unsafe_allow_html=True)
    g1,g2,g3=st.columns(3,gap="large")
    with g1:
        st.markdown("<div class='card'><div class='ctitle'>Predicho vs Real (en muestra)</div>", unsafe_allow_html=True)
        ya=d_model[TARGET].values; yp=y_pred_full
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax)
        ax.scatter(ya,yp,s=16,alpha=.4,color=ACCENT,edgecolor='white',linewidth=.4,zorder=3)
        lim=[min(ya.min(),yp.min()),max(ya.max(),yp.max())]; ax.plot(lim,lim,color=NAVY,ls='--',lw=1.5)
        ax.set_xlabel("Real"); ax.set_ylabel("Predicho"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>R² {metrics['r2']:.3f}. Cerca de la diagonal = mejor ajuste.</div></div>", unsafe_allow_html=True)
    with g2:
        st.markdown("<div class='card'><div class='ctitle'>Histograma de residuos</div>", unsafe_allow_html=True)
        res=d_model[TARGET].values-y_pred_full
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax); ax.grid(axis='x',visible=False)
        ax.hist(res,bins=18,color=NAVY,alpha=.85,edgecolor='white'); ax.axvline(0,color=ACCENT,lw=2)
        ax.set_xlabel("Residuo (real − predicho)"); ax.set_ylabel("Frecuencia"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Media ≈ {res.mean():.2f}, centrada en 0 → sin sesgo sistemático.</div></div>", unsafe_allow_html=True)
    with g3:
        st.markdown("<div class='card'><div class='ctitle'>Importancia de variables (permutación)</div>", unsafe_allow_html=True)
        idf=driver_importance(model,d_model).head(7).copy()
        idf['lab']=idf['Feature'].map(lambda c: FACTOR_LABEL.get(c,c)); idf=idf.sort_values('imp')
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax); ax.grid(axis='y',visible=False); ax.grid(axis='x',visible=True)
        ax.barh(idf['lab'],idf['imp'],color=ACCENT,zorder=3); ax.set_xlabel("Caída de R² al permutar")
        plt.tight_layout(); fig_close(fig)
        top=driver_importance(model,d_model).iloc[0]; top_lab=FACTOR_LABEL.get(top['Feature'],top['Feature'])
        st.markdown(f"<div class='note'>El mayor impulsor de la satisfacción es <b>{top_lab}</b> (diagnóstico, no prueba causal).</div></div>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{SLATE};font-size:13.5px;line-height:1.6;margin:16px 2px 12px'>"
        f"El modelo predice la <b style='color:{INK}'>satisfacción general</b> a partir de las cinco dimensiones SERVQUAL "
        f"más región, escolaridad, género y estado. Explica alrededor del <b style='color:{INK}'>{metrics['r2']*100:.0f}%</b> "
        f"de la variación, con un error típico de <b style='color:{INK}'>{s(metrics['mae'])}</b> puntos en la escala 1–5. "
        f"Las calificaciones son muy altas y concentradas, lo que limita cuánta variación puede explicar el modelo.</p>",
        unsafe_allow_html=True)
    st.markdown("<div class='limit'><div>⚠ <b>Limitación.</b> Métricas <b>en muestra</b> — entrenadas con todos los datos "
        "<b>sin división train/test</b>. Describen el ajuste a esta encuesta, no precisión futura validada. Puntajes "
        "autoreportados y sesgados al alza: resultados orientativos, no causales.</div></div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:12px;margin-top:26px;"
            f"padding-top:16px;border-top:1px solid {LINE}'>"
            "Impacto Médico · Estudio de Calidad SERVQUAL &nbsp;·&nbsp; métricas en muestra</div>", unsafe_allow_html=True)

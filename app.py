import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

# =========================
# Config
# =========================
st.set_page_config(
    page_title="DataFinscope – Piloto",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS (espacios + cards bonitas)
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .dfs-hero {
        border-radius: 18px;
        padding: 20px 22px;
        background: linear-gradient(90deg, rgba(220,235,255,0.55), rgba(222,255,241,0.55));
        border: 1px solid rgba(0,0,0,0.06);
      }
      .dfs-pill {
        display:inline-block; padding: 5px 10px; margin-right: 6px;
        border-radius: 999px; font-size: 12px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.65);
      }
      .dfs-kpi-wrap{
        display:flex; gap:14px; justify-content:center; flex-wrap:wrap; margin-top: 10px;
      }
      .dfs-kpi{
        width: 240px; max-width: 240px;
        border-radius: 16px; padding: 14px 16px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.05);
      }
      .dfs-kpi h4{ margin:0; font-size: 13px; color: rgba(0,0,0,0.65); font-weight: 600; }
      .dfs-kpi .val{ font-size: 30px; font-weight: 800; margin: 4px 0 0 0; }
      .dfs-kpi .sub{ font-size: 12px; color: rgba(0,0,0,0.55); margin-top: 2px;}
      .dfs-note{
        padding: 12px 14px; border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        background: rgba(255,255,255,0.75);
      }
      .dfs-title { margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Helpers: métricas
# =========================
def sharpe_ratio(returns, rf=0.0):
    r = pd.Series(returns).dropna()
    if len(r) < 20:
        return np.nan
    std = r.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (r.mean() - rf) / std * np.sqrt(252)

def max_drawdown(cum_curve):
    c = pd.Series(cum_curve).dropna()
    if c.empty:
        return np.nan, pd.Series(dtype=float)
    peak = c.cummax()
    dd = (c / peak) - 1
    return float(dd.min()), dd

def safe_flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance puede devolver columnas MultiIndex tipo ('Open','AAPL').
    Esto las aplana y deja nombres estándar.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        # Si viene ('Open', 'AAPL'), nos quedamos con nivel 0
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]

    # Normaliza nombres
    df.columns = [str(c).strip() for c in df.columns]
    return df

# =========================
# Perfil: reglas simples (interpretables)
# =========================
def perfil_por_score(score_mixto):
    if score_mixto < 35:
        return "Conservador"
    elif score_mixto < 70:
        return "Balanceado"
    else:
        return "Agresivo"

def recomendacion_por_perfil(perfil):
    # Más alto = más filtro = más conservador
    if perfil == "Conservador":
        return {"threshold": 0.60, "mensaje": "Entra con más filtro: menos tiempo invertido, menos ‘sustos’."}
    if perfil == "Balanceado":
        return {"threshold": 0.55, "mensaje": "Equilibrio: busca retorno con control de caídas."}
    return {"threshold": 0.50, "mensaje": "Más exposición: entra más fácil, más variación."}

# =========================
# Data + Features (mercado)
# =========================
@st.cache_data(show_spinner=False)
def download_ohlc(ticker: str, start: str) -> pd.DataFrame:
    """
    Descarga OHLCV + Adj Close (si existe) desde yfinance.
    Devuelve columnas: open, high, low, close, adj_close, volume
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False,   # evita sorpresas con Adj Close faltante
            actions=False,
            group_by="column"
        )
    except Exception:
        return pd.DataFrame()

    raw = safe_flatten_yf_columns(raw)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Mapeo robusto (según lo que exista)
    cols = {c.lower().strip(): c for c in raw.columns}

    def pick(name, fallback=None):
        if name in cols:
            return raw[cols[name]]
        return raw[fallback] if (fallback and fallback in raw.columns) else None

    o = pick("open")
    h = pick("high")
    l = pick("low")
    c = pick("close")
    v = pick("volume")

    # Adj Close a veces viene como "Adj Close"
    adj = None
    if "Adj Close" in raw.columns:
        adj = raw["Adj Close"]
    elif "adj close" in cols:
        adj = raw[cols["adj close"]]

    if c is None or v is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "open": o if o is not None else c,
        "high": h if h is not None else c,
        "low":  l if l is not None else c,
        "close": c,
        "adj_close": adj if adj is not None else c,
        "volume": v
    }, index=raw.index)

    out = out.dropna(subset=["close", "adj_close"])
    return out

def make_market_features(df_ohlc: pd.DataFrame, ma_fast=20, ma_slow=50, vol_win=20):
    full = df_ohlc.copy()
    full["ret"] = full["adj_close"].pct_change()

    full["ma20"] = full["adj_close"].rolling(ma_fast).mean()
    full["ma50"] = full["adj_close"].rolling(ma_slow).mean()

    full["dist_ma20"] = (full["adj_close"] / full["ma20"]) - 1
    full["dist_ma50"] = (full["adj_close"] / full["ma50"]) - 1

    full["ret_1"] = full["ret"].shift(1)
    full["ret_5"] = full["adj_close"].pct_change(5)
    full["ret_10"] = full["adj_close"].pct_change(10)

    full["vol20"] = full["ret"].rolling(vol_win).std()
    full["mom20"] = full["adj_close"] / full["adj_close"].shift(20) - 1

    features = ["dist_ma20", "dist_ma50", "ret_1", "ret_5", "ret_10", "vol20", "mom20"]

    full = full.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["ret", "open", "high", "low", "close"])
    X = full[features].astype(float)
    return X, full

# =========================
# Plots
# =========================
def add_invested_vrects(fig, idx, signal, opacity=0.18):
    """
    Sombrea periodos donde signal==1 usando rectángulos verticales.
    """
    s = pd.Series(signal, index=idx).fillna(0).astype(int)
    if s.empty:
        return fig

    in_pos = False
    start = None
    for t, val in s.items():
        if val == 1 and not in_pos:
            in_pos = True
            start = t
        if val == 0 and in_pos:
            end = t
            fig.add_vrect(x0=start, x1=end, fillcolor="rgba(30,144,255,1)", opacity=opacity, line_width=0)
            in_pos = False

    if in_pos and start is not None:
        fig.add_vrect(x0=start, x1=s.index[-1], fillcolor="rgba(30,144,255,1)", opacity=opacity, line_width=0)

    return fig

def plot_candles_with_signal(df, title="Precio (velas) + zonas invertido"):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Velas"
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        height=520,
        xaxis_rangeslider_visible=True
    )

    fig = add_invested_vrects(fig, df.index, df["signal_ml"], opacity=0.16)
    return fig

def plot_capital(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["capital_market_usd"], mode="lines", name="Market (Buy & Hold)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["capital_strategy_ml_usd"], mode="lines", name="Estrategia (ML)"))
    fig.update_layout(title="Capital acumulado (USD)", xaxis_title="Fecha", yaxis_title="Capital (USD)", height=420)
    return fig

def plot_drawdown(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["dd_market"], mode="lines", name="Drawdown Market"))
    fig.add_trace(go.Scatter(x=df.index, y=df["dd_ml"], mode="lines", name="Drawdown ML"))
    fig.update_layout(title="Drawdown (caídas desde el máximo)", xaxis_title="Fecha", yaxis_title="Drawdown", height=320)
    return fig

def plot_state(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["signal_ml"],
        mode="lines", fill="tozeroy",
        name="Estado (0/1)"
    ))
    fig.update_layout(
        title="Estado del inversionista (0 = efectivo, 1 = invertido)",
        xaxis_title="Fecha",
        yaxis=dict(tickvals=[0, 1], ticktext=["Efectivo", "Invertido"]),
        height=260
    )
    return fig

def plot_monthly_invested(df):
    m = df[["signal_ml"]].copy()
    m["month"] = m.index.to_period("M").astype(str)
    monthly = m.groupby("month")["signal_ml"].mean().reset_index()
    monthly["pct_invertido"] = monthly["signal_ml"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["pct_invertido"], name="% tiempo invertido"))
    fig.update_layout(
        title="Porcentaje de tiempo invertido por mes",
        xaxis_title="Mes",
        yaxis_title="% del mes invertido",
        height=320
    )
    return fig

# =========================
# Modelo
# =========================
@st.cache_resource(show_spinner=False)
def load_market_model():
    return joblib.load("modelo_ml_mercado.pkl")

model = load_market_model()

# =========================
# Session state
# =========================
if "perfil_calculado" not in st.session_state:
    st.session_state.perfil_calculado = False
if "perfil_data" not in st.session_state:
    st.session_state.perfil_data = {}
if "perfil_rec" not in st.session_state:
    st.session_state.perfil_rec = {}
if "last_run" not in st.session_state:
    st.session_state.last_run = None

# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="dfs-hero">
      <span class="dfs-pill">Piloto educativo</span>
      <span class="dfs-pill">Perfil de usuario</span>
      <span class="dfs-pill">Backtesting + ML</span>
      <h1 class="dfs-title">DataFinscope – Piloto</h1>
      <div style="font-size: 14px; color: rgba(0,0,0,0.75); margin-top:6px;">
        Esta app es un <b>prototipo</b> para explicar, con datos reales, dos cosas:
        <ul style="margin-top:6px;">
          <li><b>Perfil del usuario</b>: con 5 respuestas rápidas se clasifica el estilo (Conservador / Balanceado / Agresivo).</li>
          <li><b>Simulación histórica</b>: se compara <i>comprar y mantener</i> vs una estrategia que usa un modelo ML para decidir
              si estar <b>invertido</b> o en <b>efectivo</b>.</li>
        </ul>
        <b>Importante:</b> no es recomendación financiera. Sirve para educación y validación del prototipo.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write("")

# =========================
# Layout: izquierda (perfil) | derecha (config)
# =========================
left, right = st.columns([1.05, 0.95], gap="large")

# =========================
# Izquierda: Perfil
# =========================
with left:
    st.subheader("1) Perfil del usuario (responde rápido)")

    with st.form("perfil_form", clear_on_submit=False):
        st.markdown("**Responde 5 preguntas.** Esto ajusta el nivel de exposición recomendado (threshold) en la simulación.")

        q1 = st.slider(
            "1) ¿Qué tan cómodo te sientes usando apps o plataformas digitales? (1 = nada cómodo, 5 = muy cómodo)",
            1, 5, 3
        )
        q2 = st.slider(
            "2) Cuando lees info de inversión (videos/noticias), ¿te ayuda a decidir mejor? (1 = no, 5 = sí)",
            1, 5, 3
        )
        q3 = st.slider(
            "3) ¿Qué tanto confías en plataformas de inversión que has visto/usado? (1 = nada, 5 = mucho)",
            1, 5, 3
        )
        q4 = st.slider(
            "4) ¿Qué tanto te interesa un simulador histórico (backtesting) antes de invertir? (0 = nada, 10 = mucho)",
            0, 10, 7
        )
        q5 = st.slider(
            "5) Si existiera esta plataforma, ¿qué probabilidad hay de que la uses pronto? (1 = baja, 10 = alta)",
            1, 10, 7
        )
        pago = st.number_input(
            "Opcional: ¿Cuánto pagarías al mes por una plataforma así? (COP)",
            min_value=0, value=0, step=5000
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            calcular = st.form_submit_button("Calcular mi perfil")
        with c2:
            reset = st.form_submit_button("Reset perfil")

    if reset:
        st.session_state.perfil_calculado = False
        st.session_state.perfil_data = {}
        st.session_state.perfil_rec = {}
        st.success("Perfil reiniciado. Ahora sí, a venderle el pitch al jurado 😉")

    if calcular:
        # Scores (interpretables)
        score_con = ((q1/5) + (q2/5) + (q3/5)) / 3 * 100
        score_rie = ((q4/10) + (q5/10)) / 2 * 100
        score_mix = 0.5 * score_con + 0.5 * score_rie

        perfil = perfil_por_score(score_mix)
        rec = recomendacion_por_perfil(perfil)

        st.session_state.perfil_calculado = True
        st.session_state.perfil_data = {
            "score_conocimiento": float(score_con),
            "score_riesgo": float(score_rie),
            "score_mixto": float(score_mix),
            "perfil_final": perfil,
            "pago_mensual_cop": int(pago)
        }
        st.session_state.perfil_rec = rec

    # Mostrar SOLO si calculó
    if st.session_state.perfil_calculado:
        p = st.session_state.perfil_data
        rec = st.session_state.perfil_rec

        st.markdown(
            f"""
            <div class="dfs-kpi-wrap">
              <div class="dfs-kpi">
                <h4>Conocimiento</h4>
                <div class="val">{p['score_conocimiento']:.1f}/100</div>
                <div class="sub">Comodidad + utilidad de info + confianza.</div>
              </div>
              <div class="dfs-kpi">
                <h4>Riesgo / Interés</h4>
                <div class="val">{p['score_riesgo']:.1f}/100</div>
                <div class="sub">Interés en simulación + probabilidad de uso.</div>
              </div>
              <div class="dfs-kpi">
                <h4>Mixto</h4>
                <div class="val">{p['score_mixto']:.1f}/100</div>
                <div class="sub">Resumen para recomendar exposición.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"**Perfil final:** {p['perfil_final']}")
        st.info(f"**Recomendación (para simulación):** threshold **{rec['threshold']:.2f}**. {rec['mensaje']}")

        with st.expander("📌 ¿Qué significa este resultado? (explicación simple)", expanded=False):
            st.write(
                """
                - **Conservador**: busca menos caídas y prefiere estar menos tiempo invertido.
                - **Balanceado**: busca equilibrio entre retorno y control de riesgos.
                - **Agresivo**: tolera variación y prefiere más exposición al mercado.

                En la simulación ajustamos un “filtro” llamado **threshold**:
                - **Threshold más alto** ⇒ el modelo exige más “confianza” para invertir ⇒ entra menos.
                - **Threshold más bajo** ⇒ entra más fácil ⇒ suele estar más tiempo invertido.
                """
            )

    else:
        st.markdown('<div class="dfs-note">👈 Aún no hay perfil. Responde y presiona <b>Calcular mi perfil</b>.</div>', unsafe_allow_html=True)

# =========================
# Derecha: Config + Simulación
# =========================
with right:
    st.subheader("2) Simulación (Machine Learning de mercado)")
    st.caption("Compara Buy & Hold vs estrategia ML usando datos reales. Recomendación: usa al menos 6–12 meses de historia.")

    # Config (derecha)
    ticker = st.text_input("Ticker (ej: AAPL, MSFT, TSLA)", value="AAPL")
    start_date = st.date_input("Fecha inicio", value=date(2021, 1, 1))
    capital = st.number_input("Capital inicial (USD)", min_value=100, value=1000, step=100)
    trm = st.number_input("TRM (COP por USD)", min_value=1000, value=4000, step=50)

    # Threshold sugerido por perfil, si existe
    default_thr = 0.55
    if st.session_state.perfil_calculado:
        default_thr = float(st.session_state.perfil_rec["threshold"])

    threshold = st.slider("Threshold ML (más alto = más conservador)", 0.50, 0.70, float(default_thr), 0.01)

    run = st.button("Ejecutar backtesting ML", type="primary")

# =========================
# Run Simulation
# =========================
st.write("")
st.divider()

if run:
    with st.spinner("Descargando precios y ejecutando backtesting..."):
        df_ohlc = download_ohlc(ticker, str(start_date))

        if df_ohlc.empty or len(df_ohlc) < 120:
            st.error(
                "No se pudo descargar suficiente data (o el ticker no existe). "
                "Prueba otro ticker o una fecha de inicio más antigua (ideal 6–12 meses atrás)."
            )
            st.stop()

        X, full = make_market_features(df_ohlc)

        if X.empty or len(X) < 30:
            st.error(
                "Después de calcular medias móviles y retornos, no quedaron filas suficientes. "
                "Usa una fecha de inicio más antigua."
            )
            st.stop()

        # Proba de subida
        proba = model.predict_proba(X)[:, 1]

        full = full.loc[X.index].copy()
        full["p_up"] = proba
        full["signal_ml"] = (full["p_up"] >= threshold).astype(int)

        # Evitar lookahead: se ejecuta decisión "mañana" con señal de "hoy"
        full["strategy_ret_ml"] = full["signal_ml"].shift(1).fillna(0) * full["ret"]

        # Curvas
        full["cum_market"] = (1 + full["ret"]).cumprod()
        full["cum_strategy_ml"] = (1 + full["strategy_ret_ml"]).cumprod()

        full["capital_market_usd"] = capital * full["cum_market"]
        full["capital_strategy_ml_usd"] = capital * full["cum_strategy_ml"]

        full["capital_market_cop"] = full["capital_market_usd"] * trm
        full["capital_strategy_ml_cop"] = full["capital_strategy_ml_usd"] * trm

        # Métricas
        sh_mkt = sharpe_ratio(full["ret"])
        sh_ml = sharpe_ratio(full["strategy_ret_ml"])

        mdd_mkt, dd_mkt = max_drawdown(full["cum_market"])
        mdd_ml, dd_ml = max_drawdown(full["cum_strategy_ml"])

        full["dd_market"] = dd_mkt
        full["dd_ml"] = dd_ml

        # Guardar run para evitar “vacíos” raros en rerun
        st.session_state.last_run = full.copy()

# =========================
# Render results (si hay last_run)
# =========================
if st.session_state.last_run is not None:
    full = st.session_state.last_run

    # KPIs
    st.subheader("3) Resultados")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Capital final (Market USD)", f"${full['capital_market_usd'].iloc[-1]:,.0f}")
    k2.metric("Capital final (ML USD)", f"${full['capital_strategy_ml_usd'].iloc[-1]:,.0f}")
    k3.metric("Sharpe (Market / ML)", f"{sharpe_ratio(full['ret']):.2f} / {sharpe_ratio(full['strategy_ret_ml']):.2f}")
    k4.metric("Max Drawdown (Market / ML)", f"{max_drawdown(full['cum_market'])[0]:.1%} / {max_drawdown(full['cum_strategy_ml'])[0]:.1%}")

    with st.expander("ℹ️ ¿Cómo leer Sharpe y Max Drawdown? (súper simple)", expanded=False):
        st.write(
            """
            **Sharpe** (entre más alto, mejor) mide “retorno por unidad de riesgo”.
            - Si sube Sharpe: estás ganando “mejor” por cada susto.

            **Max Drawdown** (más cerca a 0 es mejor) es la peor caída desde un máximo.
            - Ejemplo: subes de $1000 a $1400 y luego caes a $1200 ⇒ drawdown ≈ -14.3%.
            - No es “pérdida final”, es el peor bajón durante el camino.
            """
        )

    # Semáforo de riesgo (ML)
    mdd_ml = max_drawdown(full["cum_strategy_ml"])[0]
    dd_abs = abs(mdd_ml) if not np.isnan(mdd_ml) else np.nan

    if np.isnan(dd_abs):
        st.info("No se pudo calcular drawdown (datos insuficientes o retornos nulos).")
    elif dd_abs <= 0.10:
        st.success(f"Riesgo (ML): drawdown {mdd_ml:.1%} → Zona VERDE (bajo)")
    elif dd_abs <= 0.25:
        st.warning(f"Riesgo (ML): drawdown {mdd_ml:.1%} → Zona AMARILLA (moderado)")
    else:
        st.error(f"Riesgo (ML): drawdown {mdd_ml:.1%} → Zona ROJA (alto)")

    with st.expander("🟡 ¿Qué significa “Zona amarilla / moderado”?", expanded=False):
        st.write(
            """
            Significa que **en el peor momento** la estrategia ML llegó a caer alrededor de ese porcentaje desde su máximo.
            - **Verde**: caídas pequeñas (más “amigable” para principiantes).
            - **Amarillo**: caídas medianas (toca aguantar volatilidad).
            - **Rojo**: caídas grandes (psicológicamente duro / alto riesgo).
            """
        )

    # Charts
    st.plotly_chart(plot_candles_with_signal(full, "Velas + sombreado cuando el modelo está invertido"), use_container_width=True)

    with st.expander("📌 ¿Qué significa el sombreado?", expanded=False):
        st.write(
            """
            El sombreado marca periodos donde **signal_ml = 1**, es decir:
            - el modelo estima una **probabilidad de subida (p_up)** suficiente (>= threshold)
            - y por eso decide estar **invertido**.
            """
        )

    st.plotly_chart(plot_capital(full), use_container_width=True)

    with st.expander("📌 ¿Cómo leer “Capital acumulado”?", expanded=False):
        st.write(
            """
            - **Market (Buy & Hold)**: compras el activo al inicio y no haces nada.
            - **Estrategia (ML)**: algunos días está invertida y otros días se sale a efectivo (según la señal).
            - Si la línea ML va arriba → el enfoque ML “aportó” en ese periodo.
            - Si va abajo → Buy & Hold fue mejor (normal, pasa en muchos mercados).
            """
        )

    st.plotly_chart(plot_drawdown(full), use_container_width=True)

    with st.expander("📌 ¿Qué es “Drawdown Market” vs “Drawdown ML”?", expanded=False):
        st.write(
            """
            - **Drawdown Market**: los bajones del Buy & Hold.
            - **Drawdown ML**: los bajones de la estrategia ML.
            Si el ML tiene drawdowns “menos profundos”, suele ser más tolerable para un usuario principiante.
            """
        )

    st.plotly_chart(plot_state(full), use_container_width=True)

    with st.expander("📌 ¿Qué significa “Estado del inversionista (0/1)”?", expanded=False):
        st.write(
            """
            Esto es la decisión diaria del modelo:
            - **1 = Invertido**: la estrategia está dentro del mercado.
            - **0 = Efectivo**: la estrategia se sale para reducir riesgo.

            Importante: el modelo **no adivina el futuro**; estima una probabilidad (p_up) con variables técnicas
            y decide según el threshold.
            """
        )

    st.plotly_chart(plot_monthly_invested(full), use_container_width=True)

    with st.expander("📌 ¿Cómo leer “% tiempo invertido por mes”?", expanded=False):
        st.write(
            """
            Esta barra resume qué tanto el modelo estuvo “dentro” del mercado:
            - 80% = la mayoría del mes estuvo invertido.
            - 20% = casi todo el mes estuvo en efectivo.
            Útil para explicar *comportamiento* del modelo a un usuario no técnico.
            """
        )

    # Table friendly
    st.subheader("4) Tabla (últimos registros)")
    pretty = full.copy()
    pretty["Precio (USD)"] = pretty["close"].round(2)
    pretty["Prob. de subida (p_up)"] = pretty["p_up"].round(4)
    pretty["Decisión (0/1)"] = pretty["signal_ml"].astype(int)
    pretty["Capital Market (USD)"] = pretty["capital_market_usd"].round(2)
    pretty["Capital ML (USD)"] = pretty["capital_strategy_ml_usd"].round(2)
    pretty["Capital Market (COP)"] = pretty["capital_market_cop"].round(0)
    pretty["Capital ML (COP)"] = pretty["capital_strategy_ml_cop"].round(0)

    show_cols = [
        "Precio (USD)",
        "Prob. de subida (p_up)",
        "Decisión (0/1)",
        "Capital Market (USD)",
        "Capital ML (USD)",
        "Capital Market (COP)",
        "Capital ML (COP)"
    ]

    st.dataframe(pretty[show_cols].tail(20), use_container_width=True)

    with st.expander("📌 ¿Para qué está esta tabla y qué significa cada columna?", expanded=False):
        st.write(
            """
            **Objetivo:** mostrar la “traza” del modelo día a día (transparencia). Así un usuario entiende *qué decidió* y *qué pasó*.

            - **Precio (USD):** precio del activo (cierre).
            - **Prob. de subida (p_up):** lo que cree el modelo (0 a 1) sobre que el precio suba.
            - **Decisión (0/1):**  
              - 1 = me quedo invertido  
              - 0 = me voy a efectivo
            - **Capital Market (USD/COP):** cómo evolucionaría el capital si fuera Buy & Hold.
            - **Capital ML (USD/COP):** cómo evolucionaría el capital siguiendo las decisiones ML.
            """
        )
else:
    st.markdown('<div class="dfs-note">👉 Cuando ejecutes la simulación, aquí aparecerán las gráficas y explicaciones.</div>', unsafe_allow_html=True)

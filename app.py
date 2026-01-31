# app.py
# DataFinscope – Piloto (Perfil + Simulación)
# ✅ Velas (candlestick) + sombreado cuando el modelo está invertido
# ✅ Barra mensual: % de tiempo invertido
# ✅ Explicaciones detalladas para usuarios sin experiencia
# ✅ UI amigable y organizada: izquierda (perfil), derecha (config + simulación)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import joblib

try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# Config general
# =========================
st.set_page_config(
    page_title="DataFinscope – Piloto",
    page_icon="📈",
    layout="wide",
)

# =========================
# Estilos (CSS)
# =========================
st.markdown(
    """
    <style>
      /* Layout general */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Hero */
      .dfs-hero{
        padding: 18px 18px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.06);
        background: linear-gradient(135deg, rgba(232,245,255,0.9), rgba(240,255,244,0.85));
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        margin-bottom: 14px;
      }
      .dfs-pill{
        display:inline-block;
        padding:6px 10px;
        margin-right: 6px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(0,0,0,0.06);
        color: rgba(0,0,0,0.65);
      }
      .dfs-hero h1{
        margin: 10px 0 6px 0;
        font-size: 34px;
        line-height: 1.05;
      }
      .dfs-hero p{
        margin: 6px 0;
        color: rgba(0,0,0,0.70);
        font-size: 14px;
      }
      .dfs-note{
        font-size: 13px;
        color: rgba(0,0,0,0.62);
      }

      /* Cards */
      .dfs-metric-row {
        display: flex;
        justify-content: center;
        gap: 14px;
        margin-top: 10px;
        margin-bottom: 6px;
        flex-wrap: wrap;
      }
      .dfs-card {
        width: 240px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.92);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      .dfs-card .t {
        font-size: 13px;
        color: rgba(0,0,0,0.55);
        font-weight: 800;
        margin-bottom: 6px;
      }
      .dfs-card .v {
        font-size: 36px;
        font-weight: 900;
        margin: 0;
        line-height: 1.0;
      }
      .dfs-card .s {
        font-size: 12px;
        color: rgba(0,0,0,0.55);
        margin-top: 6px;
      }
      .dfs-help {
        font-size: 12px;
        color: rgba(0,0,0,0.55);
        margin-top: 6px;
      }

      /* Botones */
      div.stButton > button:first-child {
        background: #ff4b4b;
        color: white;
        border-radius: 12px;
        border: 0px;
        padding: 0.65rem 1rem;
        font-weight: 800;
      }
      div.stButton > button:first-child:hover {
        background: #ff2f2f;
      }

      /* Secciones */
      .dfs-section-title{
        font-size: 18px;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 2px;
      }
      .dfs-subtitle{
        color: rgba(0,0,0,0.58);
        font-size: 13px;
        margin-top: 0px;
        margin-bottom: 10px;
      }

      /* Cajas informativas */
      .dfs-info{
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Utilidades
# =========================
def money_usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def money_cop(x: float) -> str:
    try:
        return f"${x:,.0f} COP"
    except Exception:
        return str(x)

def sharpe_ratio(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = ret.dropna()
    if len(ret) < 2:
        return np.nan
    mu = ret.mean()
    sd = ret.std(ddof=0)
    if sd == 0:
        return np.nan
    return float((mu / sd) * np.sqrt(periods_per_year))

def max_drawdown(cum: pd.Series) -> float:
    cum = cum.dropna()
    if len(cum) < 2:
        return np.nan
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())

def risk_zone_from_mdd(mdd_abs: float) -> tuple[str, str]:
    # mdd_abs en positivo, ej 0.138
    if np.isnan(mdd_abs):
        return ("SIN DATOS", "info")
    if mdd_abs <= 0.10:
        return ("VERDE (bajo)", "success")
    if mdd_abs <= 0.25:
        return ("AMARILLA (moderado)", "warning")
    return ("ROJA (alto)", "error")

def contiguous_segments(signal: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Devuelve lista de segmentos (inicio, fin) donde signal==1.
    """
    s = signal.fillna(0).astype(int)
    if len(s) == 0:
        return []
    changes = s.diff().fillna(0)
    starts = s.index[changes == 1].tolist()
    ends = s.index[changes == -1].tolist()

    if s.iloc[0] == 1:
        starts = [s.index[0]] + starts
    if s.iloc[-1] == 1:
        ends = ends + [s.index[-1]]

    return list(zip(starts, ends))


# =========================
# Cargar modelo ML
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

MODEL_PATH = "modelo_ml_mercado.pkl"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None


# =========================
# Descarga de precios (yfinance)
# =========================
@st.cache_data(ttl=60 * 60)
def download_prices(ticker: str, start: str):
    if yf is None:
        raise RuntimeError("No se pudo importar yfinance. Revisa requirements.txt.")
    df = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=True,   # Importante: ahora es True por defecto en yfinance
        group_by="column"
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Normalizar columnas
    # Con auto_adjust=True, 'Close' ya está ajustado (equivalente práctico a Adj Close)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            # algunos tickers pueden traer columnas faltantes
            df[c] = np.nan

    df = df[needed].copy()
    df.columns = [c.lower() for c in df.columns]  # open high low close volume
    df.index.name = "Date"
    df = df.dropna(subset=["close"]).copy()
    return df


# =========================
# Features (técnicas) para ML
# =========================
def build_features(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlc.copy()

    # retornos diarios
    df["ret_1"] = df["close"].pct_change()

    # ventanas
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # distancia a medias móviles (qué tan arriba/abajo del promedio)
    df["dist_ma20"] = (df["close"] / df["ma20"]) - 1.0
    df["dist_ma50"] = (df["close"] / df["ma50"]) - 1.0

    # retornos acumulados a varios horizontes (aprox)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # momentum 20 días (cómo viene vs hace 20 días)
    df["mom20"] = (df["close"] / df["close"].shift(20)) - 1.0

    # volatilidad rolling 20 días (qué tan “movido” está)
    df["vol20"] = df["ret_1"].rolling(20).std()

    feats = ["dist_ma20", "dist_ma50", "ret_1", "ret_5", "ret_10", "vol20", "mom20"]
    return df, feats


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predicción robusta:
    - Asegura orden de columnas si el modelo lo guarda
    - Convierte a float y reemplaza inf
    - Maneja NaNs (pero idealmente X ya llega limpio)
    """
    X2 = X.copy()
    X2 = X2.replace([np.inf, -np.inf], np.nan)

    # Si el modelo conoce nombres originales:
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        # si faltan columnas, crear con nan
        for c in cols:
            if c not in X2.columns:
                X2[c] = np.nan
        X2 = X2[cols]

    # Convertir a float
    X2 = X2.astype(float)

    # Relleno simple (para producción educativa):
    # - mediana por columna (evita romper transform)
    for c in X2.columns:
        if X2[c].isna().any():
            X2[c] = X2[c].fillna(X2[c].median())

    proba = model.predict_proba(X2)[:, 1]
    return proba


# =========================
# Backtesting ML (con threshold)
# =========================
def run_backtest_ml(
    df_ohlc: pd.DataFrame,
    model,
    feats: list[str],
    threshold: float,
    capital_inicial_usd: float,
    trm: float
):
    df, feats = build_features(df_ohlc)
    df = df.dropna(subset=feats + ["ret_1", "close"]).copy()
    if len(df) < 200:
        return None, "Necesitas más historia (ideal: 6–12 meses o más) para que existan MA50, vol20, etc."

    # Predicción
    X = df[feats].copy()
    df["p_up"] = predict_proba_safe(model, X)

    # Señal ML (decisión del modelo)
    df["signal_ml"] = (df["p_up"] >= threshold).astype(int)

    # Para evitar “mirar el futuro”, aplicamos la decisión al día siguiente
    df["position_ml"] = df["signal_ml"].shift(1).fillna(0).astype(int)

    # Retorno estrategia
    df["strategy_ml_ret"] = df["position_ml"] * df["ret_1"]

    # Buy & Hold
    df["cum_market"] = (1 + df["ret_1"]).cumprod()
    df["cum_strategy_ml"] = (1 + df["strategy_ml_ret"]).cumprod()

    df["capital_market_usd"] = capital_inicial_usd * df["cum_market"]
    df["capital_strategy_ml_usd"] = capital_inicial_usd * df["cum_strategy_ml"]

    df["capital_market_cop"] = df["capital_market_usd"] * trm
    df["capital_strategy_ml_cop"] = df["capital_strategy_ml_usd"] * trm

    # Drawdown series
    peak_m = df["cum_market"].cummax()
    df["dd_market"] = (df["cum_market"] / peak_m) - 1.0

    peak_s = df["cum_strategy_ml"].cummax()
    df["dd_strategy_ml"] = (df["cum_strategy_ml"] / peak_s) - 1.0

    # Métricas
    sh_market = sharpe_ratio(df["ret_1"])
    sh_ml = sharpe_ratio(df["strategy_ml_ret"])

    mdd_market = float(df["dd_market"].min())
    mdd_ml = float(df["dd_strategy_ml"].min())

    out = {
        "df": df,
        "sh_market": sh_market,
        "sh_ml": sh_ml,
        "mdd_market": mdd_market,
        "mdd_ml": mdd_ml,
        "cap_market_final": float(df["capital_market_usd"].iloc[-1]),
        "cap_ml_final": float(df["capital_strategy_ml_usd"].iloc[-1]),
    }
    return out, None


# =========================
# UI: HERO
# =========================
st.markdown(
    """
    <div class="dfs-hero">
      <span class="dfs-pill">Piloto educativo</span>
      <span class="dfs-pill">Perfil + Simulación</span>
      <span class="dfs-pill">Backtesting real</span>

      <h1>DataFinscope – Piloto</h1>

      <p>
        Esta app te ayuda a <b>entender tu perfil</b> y luego te muestra una <b>simulación histórica</b>
        comparando dos caminos:
        <br/>✅ <b>Market (Buy & Hold)</b>: compras el activo y lo mantienes.
        <br/>✅ <b>Estrategia ML</b>: el modelo decide si estar “invertido” o “en efectivo”.
      </p>

      <p class="dfs-note">
        <b>Cómo usarla (modo fácil):</b><br/>
        1) Responde 5 preguntas → obtienes un perfil (Conservador / Balanceado / Agresivo).<br/>
        2) Configuras ticker, fecha y capital → ejecutas backtesting.<br/>
        3) Ves resultados: Sharpe, Drawdown, gráficas y tabla explicativa.
      </p>

      <p class="dfs-note">
        ⚠️ <b>No es recomendación financiera</b>. Es un piloto académico para demostrar ciencia de datos, analítica y visualización.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.error("No pude cargar `modelo_ml_mercado.pkl`. Verifica que esté en el repo junto a `app.py`.")
    st.stop()

if yf is None:
    st.error("No pude importar yfinance. Revisa requirements.txt (debe incluir yfinance).")
    st.stop()


# =========================
# Layout principal: 2 columnas
# =========================
left, right = st.columns([1.05, 1.0], gap="large")

# =========================
# Columna izquierda: Perfil
# =========================
with left:
    st.markdown('<div class="dfs-section-title">1) Perfil del usuario</div>', unsafe_allow_html=True)
    st.markdown('<div class="dfs-subtitle">Responde rápido. Esto ajusta la recomendación de exposición (threshold) para la simulación.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="dfs-info">', unsafe_allow_html=True)

        with st.expander("🧩 Responder (5 preguntas rápidas)", expanded=True):
            q1 = st.slider(
                "1) ¿Qué tan cómodo te sientes usando apps o herramientas digitales? (1 = nada cómodo, 5 = muy cómodo)",
                1, 5, 3
            )
            q2 = st.slider(
                "2) ¿Qué tanto te ayuda la información de inversión a tomar decisiones? (1 = no ayuda, 5 = ayuda mucho)",
                1, 5, 3
            )
            q3 = st.slider(
                "3) ¿Qué tan probable es que uses una plataforma educativa como esta? (1 = nada probable, 10 = muy probable)",
                1, 10, 7
            )
            q4 = st.slider(
                "4) ¿Cuánta confianza te dan las plataformas/apps de inversión que has visto o usado? (1 = nada, 5 = mucha)",
                1, 5, 3
            )
            q5 = st.slider(
                "5) ¿Qué tanto te interesa un simulador histórico (backtesting) para aprender? (0 = nada, 10 = muchísimo)",
                0, 10, 7
            )

            pago_cop = st.number_input(
                "Opcional: ¿Cuánto pagarías al mes por una plataforma así? (COP)",
                min_value=0,
                value=15000,
                step=1000
            )

            btn_profile = st.button("Calcular mi perfil")

        # Scores simples y entendibles (educativos)
        def to_100(x, minv, maxv):
            return 100 * (x - minv) / (maxv - minv)

        score_conocimiento = np.mean([to_100(q1, 1, 5), to_100(q2, 1, 5)])  # comodidad + utilidad
        score_riesgo = np.mean([to_100(q3, 1, 10), to_100(q5, 0, 10), to_100(q4, 1, 5)])  # interés + confianza
        score_mixto = 0.5 * score_conocimiento + 0.5 * score_riesgo

        # Perfil final (reglas claras)
        if score_mixto >= 70:
            perfil_final = "Agresivo"
            reco_threshold = 0.50
            perfil_msg = "Toleras más variación y buscas mayor exposición."
        elif score_mixto >= 45:
            perfil_final = "Balanceado"
            reco_threshold = 0.55
            perfil_msg = "Equilibrio entre retorno y control de caídas."
        else:
            perfil_final = "Conservador"
            reco_threshold = 0.60
            perfil_msg = "Prefieres menos sustos; el modelo entra con más filtro."

        # Guardar en session
        if btn_profile:
            st.session_state["perfil"] = {
                "score_conocimiento": float(score_conocimiento),
                "score_riesgo": float(score_riesgo),
                "score_mixto": float(score_mixto),
                "perfil_final": perfil_final,
                "reco_threshold": reco_threshold,
                "pago_cop": int(pago_cop),
            }

        p = st.session_state.get("perfil", {
            "score_conocimiento": float(score_conocimiento),
            "score_riesgo": float(score_riesgo),
            "score_mixto": float(score_mixto),
            "perfil_final": perfil_final,
            "reco_threshold": reco_threshold,
            "pago_cop": int(pago_cop),
        })

        st.markdown(
            f"""
            <div class="dfs-metric-row">
              <div class="dfs-card">
                <div class="t">Conocimiento</div>
                <div class="v">{p['score_conocimiento']:.1f}/100</div>
                <div class="s">Comodidad + claridad con herramientas.</div>
              </div>
              <div class="dfs-card">
                <div class="t">Riesgo / Interés</div>
                <div class="v">{p['score_riesgo']:.1f}/100</div>
                <div class="s">Interés en simulación + confianza.</div>
              </div>
              <div class="dfs-card">
                <div class="t">Mixto</div>
                <div class="v">{p['score_mixto']:.1f}/100</div>
                <div class="s">Resumen general para recomendar exposición.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"Perfil final: **{p['perfil_final']}**")

        # Recomendación de threshold (educativa)
        if p["perfil_final"] == "Agresivo":
            st.info(f"Recomendación: threshold **{p['reco_threshold']:.2f}**. Entra más fácil al mercado (más exposición, más variación).")
        elif p["perfil_final"] == "Balanceado":
            st.info(f"Recomendación: threshold **{p['reco_threshold']:.2f}**. Balance entre exposición y control de caídas.")
        else:
            st.info(f"Recomendación: threshold **{p['reco_threshold']:.2f}**. Entra con más filtro (menos exposición, menos sustos).")

        with st.expander("🧠 ¿Qué significa este resultado? (explicación simple)"):
            st.markdown(
                f"""
                **Tu perfil: {p['perfil_final']}**

                - **Conservador:** prioriza estabilidad. Prefiere evitar caídas fuertes.
                - **Balanceado:** busca equilibrio. Tolera algo de variación.
                - **Agresivo:** tolera más variación con tal de buscar más retorno.

                **¿Qué cambia en la simulación?**  
                Ajustamos el **threshold** del modelo.

                - **Threshold más alto** ⇒ el modelo exige más “confianza” para entrar ⇒ suele estar menos tiempo invertido.  
                - **Threshold más bajo** ⇒ entra más fácil ⇒ suele estar más tiempo invertido.

                ✅ Importante: esto es educativo. No “predice tu futuro”, solo ajusta la experiencia del piloto.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Columna derecha: Config + simulación
# =========================
with right:
    st.markdown('<div class="dfs-section-title">2) Simulación (Machine Learning de mercado)</div>', unsafe_allow_html=True)
    st.markdown('<div class="dfs-subtitle">Compara Buy & Hold vs Estrategia ML, usando datos reales. Recomendación: usa al menos 6–12 meses de historia.</div>', unsafe_allow_html=True)

    # Configuración
    with st.container():
        st.markdown('<div class="dfs-info">', unsafe_allow_html=True)

        cA, cB = st.columns(2)
        with cA:
            ticker = st.text_input("Ticker (ej: AAPL, MSFT, TSLA)", value="AAPL")
            start_date = st.date_input("Fecha inicio", value=date(2021, 1, 1), min_value=date(1990, 1, 1))
        with cB:
            capital_inicial_usd = st.number_input("Capital inicial (USD)", min_value=100.0, value=1000.0, step=100.0)
            trm = st.number_input("TRM (COP por USD)", min_value=1000.0, value=4000.0, step=50.0)

        # Threshold recomendado por perfil (pero editable)
        default_th = float(p.get("reco_threshold", 0.55))
        threshold = st.slider(
            "Threshold ML (más alto = más conservador)",
            0.45, 0.70, float(default_th), 0.01
        )

        run_btn = st.button("Ejecutar backtesting ML")

        st.markdown("</div>", unsafe_allow_html=True)

    # Ejecutar
    if run_btn:
        with st.spinner("Descargando datos y ejecutando simulación…"):
            df_ohlc = download_prices(ticker, str(start_date))
            if df_ohlc.empty:
                st.error("No pude descargar datos. Revisa el ticker o prueba otra fecha.")
                st.stop()

            out, err = run_backtest_ml(
                df_ohlc=df_ohlc,
                model=model,
                feats=["dist_ma20", "dist_ma50", "ret_1", "ret_5", "ret_10", "vol20", "mom20"],
                threshold=threshold,
                capital_inicial_usd=float(capital_inicial_usd),
                trm=float(trm)
            )
            if err:
                st.error(err)
                st.stop()

            st.session_state["bt"] = out

    out = st.session_state.get("bt", None)
    if out is None:
        st.info("Configura y ejecuta el backtesting para ver resultados.")
        st.stop()

    df = out["df"].copy()
    sh_market, sh_ml = out["sh_market"], out["sh_ml"]
    mdd_market, mdd_ml = out["mdd_market"], out["mdd_ml"]
    cap_m, cap_ml = out["cap_market_final"], out["cap_ml_final"]

    # =========================
    # KPIs
    # =========================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Capital final (Market USD)", money_usd(cap_m))
    k2.metric("Capital final (ML USD)", money_usd(cap_ml))
    k3.metric("Sharpe (Market / ML)", f"{sh_market:.2f} / {sh_ml:.2f}")
    k4.metric("Max Drawdown (Market / ML)", f"{mdd_market:.1%} / {mdd_ml:.1%}")

    # Semáforo de riesgo (basado en drawdown ML)
    zone, zone_kind = risk_zone_from_mdd(abs(mdd_ml))
    msg = f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona **{zone}**"
    if zone_kind == "success":
        st.success(msg)
    elif zone_kind == "warning":
        st.warning(msg)
    else:
        st.error(msg)

    # Explicación Sharpe + Drawdown
    with st.expander("🧠 ¿Qué significan Sharpe y Max Drawdown? (explicado fácil)"):
        st.markdown(
            """
            ### Sharpe (Market / ML)
            **Idea simple:** “¿Cuánto retorno obtienes por cada unidad de riesgo?”  
            - Si **Sharpe es más alto**, en general la estrategia fue **más eficiente** (mejor retorno para el nivel de variación).
            - Regla mental rápida:
              - **< 0.5**: flojo (mucho riesgo para lo que devuelve)
              - **0.5 – 1.0**: aceptable
              - **> 1.0**: bueno (para piloto educativo)

            **Market** = comprar y mantener.  
            **ML** = el modelo decide si estar invertido o en efectivo con `p_up` y el `threshold`.

            ### Max Drawdown (Market / ML)
            **Idea simple:** “¿Cuál fue la peor caída desde un máximo?”  
            No es la pérdida final; es el peor susto del camino.

            Ejemplo: subes de **$1,000 → $1,400** y luego bajas a **$1,100**  
            → drawdown ≈ **-21.4%**.

            ✅ Un drawdown menor suele ser más cómodo para principiantes.
            """
        )

    with st.expander("🚦 ¿Qué significa esta zona de riesgo?"):
        st.markdown(
            f"""
            Tu estrategia ML tuvo un **Max Drawdown de {mdd_ml:.1%}**.

            **Qué significa en la práctica:**  
            En el peor momento, tu capital estuvo aproximadamente **{abs(mdd_ml)*100:.1f}%** por debajo de su máximo histórico.

            **Semáforo (interpretación educativa):**
            - 🟢 **Verde (≤ 10%)**: caídas pequeñas, más cómodo.
            - 🟡 **Amarillo (10%–25%)**: caídas moderadas; requiere paciencia.
            - 🔴 **Rojo (> 25%)**: caídas fuertes; emocionalmente más difícil.

            ✅ Importante: “amarillo” no significa “malo”, significa “hay sustos moderados”.
            """
        )

    # =========================
    # A) Candlestick + sombreado invertido
    # =========================
    st.markdown("### A) Precio (velas) + cuándo el modelo estuvo invertido")
    st.caption("Sombreado verde = el modelo estaba **invertido** (expuesto al mercado). Sin sombra = estaba **en efectivo**.")

    df_plot = df.copy()
    # Para candlestick necesitamos OHLC en df_plot: ya está en df (proviene de ohlc), pero tras features se conserva
    # Asegurar columnas
    for c in ["open", "high", "low", "close"]:
        if c not in df_plot.columns:
            df_plot[c] = np.nan

    fig_candle = go.Figure()

    fig_candle.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="Precio (velas)"
        )
    )

    # Sombreado por segmentos invertidos (position_ml == 1)
    segments = contiguous_segments(df_plot["position_ml"])
    for (s0, s1) in segments:
        fig_candle.add_vrect(
            x0=s0, x1=s1,
            fillcolor="rgba(0, 200, 0, 0.12)",
            line_width=0
        )

    # Entradas / salidas (markers)
    pos = df_plot["position_ml"].fillna(0).astype(int)
    entry = (pos.diff() == 1)
    exit_ = (pos.diff() == -1)

    fig_candle.add_trace(
        go.Scatter(
            x=df_plot.index[entry],
            y=df_plot.loc[entry, "close"],
            mode="markers",
            name="Entrada (ML)",
            marker=dict(symbol="triangle-up", size=10)
        )
    )
    fig_candle.add_trace(
        go.Scatter(
            x=df_plot.index[exit_],
            y=df_plot.loc[exit_, "close"],
            mode="markers",
            name="Salida (ML)",
            marker=dict(symbol="triangle-down", size=10)
        )
    )

    fig_candle.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig_candle, use_container_width=True)

    with st.expander("📌 ¿Cómo leer este gráfico? (modo súper fácil)"):
        st.markdown(
            """
            **Qué muestra:** el precio real del activo en forma de velas (como TradingView).

            **Sombra verde:** días en los que el modelo estuvo **invertido** (asumimos exposición al mercado).  
            **Sin sombra:** días en los que el modelo estuvo **en efectivo** (sale para reducir riesgo).

            **Triángulos:**
            - ▲ Entrada: el modelo decidió pasar de efectivo → invertido.
            - ▼ Salida: el modelo decidió pasar de invertido → efectivo.

            ✅ Esto ayuda a entender *cuándo* el modelo toma decisiones, no solo el resultado final.
            """
        )

    # =========================
    # B) Capital acumulado + Drawdown
    # =========================
    st.markdown("### B) Capital acumulado (USD) y Drawdown (caídas)")
    colB1, colB2 = st.columns(2)

    with colB1:
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(x=df.index, y=df["capital_market_usd"], mode="lines", name="Market (Buy & Hold)"))
        fig_cap.add_trace(go.Scatter(x=df.index, y=df["capital_strategy_ml_usd"], mode="lines", name="Estrategia (ML)"))
        fig_cap.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Fecha",
            yaxis_title="Capital (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        with st.expander("📌 ¿Cómo leer 'Capital acumulado'?"):
            st.markdown(
                """
                **Qué muestra:** cómo habría evolucionado tu dinero en el tiempo, según el camino elegido.

                - **Market (Buy & Hold):** compras el activo y lo mantienes.
                - **Estrategia (ML):** el modelo decide cuándo estar invertido o cuándo estar en efectivo.

                **Interpretación rápida:**
                - Si la línea ML está arriba → ML aportó valor en ese periodo.
                - Si está abajo → el mercado fue mejor en ese tramo (es normal que pase).
                """
            )

    with colB2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df.index, y=df["dd_market"], mode="lines", name="Drawdown Market"))
        fig_dd.add_trace(go.Scatter(x=df.index, y=df["dd_strategy_ml"], mode="lines", name="Drawdown ML"))
        fig_dd.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Fecha",
            yaxis_title="Drawdown (0 a negativo)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        with st.expander("📉 ¿Cómo leer Drawdown? (sustos del camino)"):
            st.markdown(
                """
                **Drawdown = caída desde el máximo (no es pérdida final).**

                Ejemplo: subes de 1,000 a 1,400 y bajas a 1,100 → drawdown ≈ -21.4%.

                **Cómo usarlo:**
                - Si ML tiene drawdown menos profundo (más cerca de 0) → suele controlar caídas.
                - Si es más profundo → puede ser más difícil emocionalmente.

                ✅ Para principiantes, normalmente se prefiere drawdown menor.
                """
            )

    # =========================
    # C) % tiempo invertido mensual (barra)
    # =========================
    st.markdown("### C) ¿Cuánto tiempo estuvo invertido el modelo? (% mensual)")
    st.caption("Esto es muy fácil de entender: 100% = todo el mes invertido, 0% = todo el mes en efectivo.")

    monthly_on = (df["position_ml"].resample("M").mean() * 100).round(1)
    monthly_df = monthly_on.reset_index()
    monthly_df.columns = ["Mes", "%_tiempo_invertido"]

    fig_month = px.bar(
        monthly_df,
        x="Mes",
        y="%_tiempo_invertido",
        title="Porcentaje mensual de tiempo invertido (ML)",
    )
    fig_month.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Mes",
        yaxis_title="% del mes invertido"
    )
    st.plotly_chart(fig_month, use_container_width=True)

    with st.expander("📌 ¿Cómo interpretar esta barra mensual?"):
        st.markdown(
            """
            **Qué significa:** qué tanto tiempo el modelo estuvo expuesto al mercado ese mes.

            - **80%**: el modelo estuvo invertido la mayor parte del mes.
            - **20%**: estuvo casi todo el mes en efectivo (más conservador).

            ✅ Útil para explicar el “estilo” de la estrategia:  
            si se expone mucho (más retorno potencial, más variación) o si se protege más (menos sustos).
            """
        )

    # =========================
    # D) Tabla últimos registros (explicación completa)
    # =========================
    st.markdown("### D) Tabla (últimos registros)")
    df_table = df.copy()

    # Mostrar columnas amigables
    table = pd.DataFrame({
        "Date": df_table.index,
        "Precio (USD)": df_table["close"].round(2),
        "Prob. subida (p_up)": df_table["p_up"].round(4),
        "Decisión (0/1)": df_table["position_ml"].astype(int),
        "Capital Market (USD)": df_table["capital_market_usd"].round(2),
        "Capital ML (USD)": df_table["capital_strategy_ml_usd"].round(2),
        "Capital Market (COP)": df_table["capital_market_cop"].round(0),
        "Capital ML (COP)": df_table["capital_strategy_ml_cop"].round(0),
    }).tail(12)

    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("📋 ¿Qué es esta tabla y para qué sirve? (columna por columna)"):
        st.markdown(
            """
            **Objetivo de la tabla:**  
            Mostrar el “detalle operativo” de la simulación: qué probabilidad calculó el modelo, qué decisión tomó (0/1) y cómo impactó el capital.

            **Columnas:**
            - **Precio (USD):** precio del activo ese día.
            - **Prob. subida (p_up):** probabilidad (0–1) estimada por el modelo de que el precio suba.
              - Ej: 0.56 ≈ 56% (según señales técnicas).
            - **Decisión (0/1):**
              - **1 = Invertido:** el modelo decide estar en el mercado.
              - **0 = Efectivo:** el modelo decide salir para reducir riesgo.
            - **Capital Market (USD):** capital si solo compraras y mantuvieras.
            - **Capital ML (USD):** capital siguiendo decisiones del modelo.
            - **Capital Market (COP):** conversión a pesos con TRM.
            - **Capital ML (COP):** conversión a pesos con TRM.

            ✅ Esta tabla es clave para el jurado: demuestra que el resultado no es “magia”, sino decisiones auditable.
            """
        )

    # =========================
    # Glosario
    # =========================
    with st.expander("📚 Glosario rápido (para principiantes)"):
        st.markdown(
            """
            - **Backtesting:** probar una estrategia en el pasado para ver cómo habría funcionado.
            - **Buy & Hold:** comprar y mantener sin hacer cambios.
            - **Sharpe:** retorno por unidad de riesgo (eficiencia).
            - **Drawdown:** caída desde el máximo (peor susto).
            - **p_up:** probabilidad de subida estimada por el modelo (0 a 1).
            - **Threshold:** filtro de confianza para entrar (más alto = más conservador).
            - **Invertido vs Efectivo:** estar expuesto al mercado vs salir temporalmente para reducir caídas.
            """
        )

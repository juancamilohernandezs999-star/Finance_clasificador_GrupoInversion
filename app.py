# app.py
# DataFinscope – Piloto (Perfil + Simulación ML)
# Nota: Prototipo educativo. No es recomendación financiera.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import date

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px


# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="DataFinscope – Piloto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
/* Padding arriba para que NO se vea cortado */
.block-container { padding-top: 2.0rem !important; padding-bottom: 2rem; }

/* Evita que queden “huecos” raros y centra el contenido */
.main .block-container { max-width: 1250px; margin: 0 auto; }

/* Tarjetas (cards) */
.dfs-card{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 25px rgba(15,23,42,0.06);
}
.dfs-card h4{ margin:0; font-size: 0.9rem; color: #64748b; font-weight: 600; }
.dfs-card .big{ font-size: 2.0rem; font-weight: 800; color:#0f172a; margin-top:6px; }
.dfs-card .sub{ margin-top: 6px; font-size: 0.85rem; color:#475569; }

/* Badge */
.dfs-badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  background: rgba(2,132,199,0.08);
  color: #075985;
  border: 1px solid rgba(2,132,199,0.15);
  margin-right: 8px;
}

/* Header “hero” */
.dfs-hero{
  border-radius: 18px;
  padding: 22px 22px;
  background: linear-gradient(90deg, rgba(226,232,240,0.65), rgba(224,242,254,0.55));
  border: 1px solid rgba(15,23,42,0.06);
}
.dfs-hero h1{
  margin: 6px 0 8px 0;
  font-size: 2.2rem;
  line-height: 1.15;
  color:#0f172a;
}
.dfs-hero p{
  margin: 0;
  font-size: 0.95rem;
  color:#334155;
}

/* Centrar filas de cards */
.dfs-center-row{
  display:flex;
  justify-content:center;
  gap: 18px;
  flex-wrap: wrap;
}

/* Separadores suaves */
hr { border: none; border-top: 1px solid rgba(15,23,42,0.08); margin: 18px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers numéricos / finanzas
# -----------------------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def sharpe_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    """Sharpe simple con retornos diarios (sin annualizar para no marear al usuario)."""
    r = daily_returns.dropna()
    if len(r) < 5:
        return np.nan
    excess = r - rf_daily
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd)


def max_drawdown_from_curve(curve: pd.Series) -> float:
    """Max drawdown de una curva de capital (ej. cum_strategy)."""
    c = curve.dropna()
    if len(c) < 5:
        return np.nan
    peak = c.cummax()
    dd = (c / peak) - 1.0
    return float(dd.min())


def risk_zone_from_mdd(mdd: float):
    """
    Clasifica el riesgo según el Max Drawdown (mdd es negativo).
    - Verde: 0% a -10%
    - Amarilla: -10% a -20%
    - Roja: < -20%
    """
    if np.isnan(mdd):
        return ("SIN DATOS", "info")
    x = abs(mdd)
    if x <= 0.10:
        return ("VERDE (bajo)", "success")
    if x <= 0.20:
        return ("AMARILLA (moderado)", "warning")
    return ("ROJA (alto)", "error")


def format_currency_usd(x):
    return f"${x:,.0f}"


def format_currency_cop(x):
    return f"${x:,.0f} COP"


# -----------------------------
# Descarga robusta de precios
# -----------------------------
@st.cache_data(show_spinner=False)
def download_prices(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Descarga OHLCV con yfinance y normaliza columnas:
    date index, columns: open, high, low, close, volume
    """
    # yfinance a veces devuelve multiindex si tickers list o ciertos modos.
    df = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,     # para tener OHLC “real”
        group_by="column",
        threads=True
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Si viene con MultiIndex (col, ticker), lo aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        # nos quedamos con el nivel 0 (Open/High/Low/Close/Adj Close/Volume)
        # o tomamos la 2da dimensión si aplica
        df.columns = [c[0] for c in df.columns]

    # Normalizamos a minúsculas
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(set(df.columns)):
        # Intento alterno: a veces Close y Volume sí están pero Open/High/Low no.
        # En ese caso, al menos devolvemos close/volume y creamos OHLC “proxy”.
        if "close" in df.columns:
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = np.nan

    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df


# -----------------------------
# Features + ML + Backtesting
# -----------------------------
def add_features(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlc.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # Distancia vs medias: (precio - MA)/MA
    df["dist_ma20"] = (df["close"] - df["ma20"]) / df["ma20"]
    df["dist_ma50"] = (df["close"] - df["ma50"]) / df["ma50"]

    # Volatilidad rolling 20
    df["vol20"] = df["ret_1"].rolling(20).std()

    # Momentum 20: retorno acumulado 20 días (aprox)
    df["mom20"] = (1 + df["ret_1"]).rolling(20).apply(np.prod, raw=True) - 1

    return df


def run_ml_backtest(
    model,
    ticker: str,
    user_start_date: date,
    capital_inicial_usd: float,
    trm_cop_usd: float,
    threshold: float,
) -> dict:
    """
    Devuelve dict con:
      df (con columnas finales),
      métricas,
      figuras plotly
    """
    # Para calcular MA50, vol20 etc., pedimos historia extra (buffer)
    start_dt = pd.to_datetime(user_start_date)
    fetch_start = (start_dt - pd.Timedelta(days=220)).date()

    df_ohlc = download_prices(ticker, str(fetch_start))
    if df_ohlc.empty:
        return {"error": "No se pudieron descargar datos. Revisa el ticker o la fecha."}

    df = add_features(df_ohlc)

    # Target conceptual: probabilidad de que el próximo día sea positivo (p_up)
    # OJO: en backtesting NO podemos “ver” el futuro para decidir hoy.
    # El modelo estima p_up(t) con features de t.
    feature_cols = ["dist_ma20", "dist_ma50", "ret_1", "ret_5", "ret_10", "vol20", "mom20"]

    df_model = df.dropna(subset=feature_cols).copy()
    if df_model.empty or len(df_model) < 80:
        return {"error": "Muy pocos datos después de calcular indicadores. Usa una fecha de inicio más antigua."}

    # Solo el rango desde user_start_date en adelante (pero ya con indicadores listos)
    df_model = df_model[df_model.index >= start_dt].copy()
    if len(df_model) < 40:
        return {"error": "Rango muy corto. Usa al menos 6–12 meses para resultados más estables."}

    X = df_model[feature_cols].astype(float)

    # Probabilidad de subida (clase 1)
    proba = model.predict_proba(X)[:, 1]
    df_model["p_up"] = proba

    # Señal del modelo: 1 = invertido, 0 = efectivo
    df_model["signal_ml"] = (df_model["p_up"] >= threshold).astype(int)

    # IMPORTANTE: decisión de hoy se ejecuta “mañana” (shift(1)) para evitar look-ahead.
    df_model["pos_ml"] = df_model["signal_ml"].shift(1).fillna(0).astype(int)

    # Retornos
    df_model["ret"] = df_model["close"].pct_change().fillna(0)

    # Estrategias:
    df_model["strategy_ml_ret"] = df_model["pos_ml"] * df_model["ret"]
    df_model["cum_market"] = (1 + df_model["ret"]).cumprod()
    df_model["cum_ml"] = (1 + df_model["strategy_ml_ret"]).cumprod()

    # Capital
    df_model["capital_market_usd"] = capital_inicial_usd * df_model["cum_market"]
    df_model["capital_ml_usd"] = capital_inicial_usd * df_model["cum_ml"]
    df_model["capital_market_cop"] = df_model["capital_market_usd"] * trm_cop_usd
    df_model["capital_ml_cop"] = df_model["capital_ml_usd"] * trm_cop_usd

    # Métricas
    sh_market = sharpe_ratio(df_model["ret"])
    sh_ml = sharpe_ratio(df_model["strategy_ml_ret"])
    mdd_market = max_drawdown_from_curve(df_model["cum_market"])
    mdd_ml = max_drawdown_from_curve(df_model["cum_ml"])

    # Drawdown series (para gráfica)
    peak_m = df_model["cum_market"].cummax()
    df_model["dd_market"] = (df_model["cum_market"] / peak_m) - 1.0
    peak_s = df_model["cum_ml"].cummax()
    df_model["dd_ml"] = (df_model["cum_ml"] / peak_s) - 1.0

    # % tiempo invertido mensual
    monthly = (
        df_model[["pos_ml"]]
        .resample("M")
        .mean()
        .rename(columns={"pos_ml": "pct_invertido"})
    )
    monthly["pct_invertido"] = monthly["pct_invertido"] * 100

    # Figuras
    fig_candles = make_candlestick_with_invest_shading(df_model)
    fig_capital = make_capital_chart(df_model)
    fig_drawdown = make_drawdown_chart(df_model)
    fig_state = make_state_chart(df_model)
    fig_monthly = make_monthly_invested_bar(monthly)

    return {
        "df": df_model,
        "metrics": {
            "capital_market_usd": float(df_model["capital_market_usd"].iloc[-1]),
            "capital_ml_usd": float(df_model["capital_ml_usd"].iloc[-1]),
            "sh_market": float(sh_market) if not np.isnan(sh_market) else np.nan,
            "sh_ml": float(sh_ml) if not np.isnan(sh_ml) else np.nan,
            "mdd_market": float(mdd_market) if not np.isnan(mdd_market) else np.nan,
            "mdd_ml": float(mdd_ml) if not np.isnan(mdd_ml) else np.nan,
        },
        "figs": {
            "candles": fig_candles,
            "capital": fig_capital,
            "drawdown": fig_drawdown,
            "state": fig_state,
            "monthly": fig_monthly,
        },
        "monthly": monthly,
    }


def make_candlestick_with_invest_shading(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Velas (precio)",
            increasing_line_width=1,
            decreasing_line_width=1,
        )
    )

    # Sombreado donde pos_ml == 1 (invertido)
    # Creamos rangos continuos para no poner 10,000 rectángulos.
    invested = df["pos_ml"].fillna(0).astype(int).values
    idx = df.index.to_list()

    spans = []
    in_span = False
    start = None
    for i in range(len(invested)):
        if invested[i] == 1 and not in_span:
            in_span = True
            start = idx[i]
        if invested[i] == 0 and in_span:
            end = idx[i]
            spans.append((start, end))
            in_span = False
    if in_span:
        spans.append((start, idx[-1]))

    for (a, b) in spans:
        fig.add_vrect(
            x0=a, x1=b,
            fillcolor="rgba(34,197,94,0.10)",  # verde suave
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        title="Precio (velas) + sombreado cuando el modelo está INVERTIDO",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def make_capital_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["capital_market_usd"], mode="lines", name="Market (Buy & Hold)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["capital_ml_usd"], mode="lines", name="Estrategia (ML)"))
    fig.update_layout(
        title="Capital acumulado (USD)",
        xaxis_title="Fecha",
        yaxis_title="Capital (USD)",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["dd_market"], mode="lines", name="Drawdown Market"))
    fig.add_trace(go.Scatter(x=df.index, y=df["dd_ml"], mode="lines", name="Drawdown ML"))
    fig.update_layout(
        title="Drawdown (caídas desde el máximo histórico)",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (negativo = caída)",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_state_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["pos_ml"],
            mode="lines",
            line_shape="hv",
            name="Estado (ML)",
            fill="tozeroy",
            opacity=0.35,
        )
    )
    fig.update_layout(
        title="Estado del inversionista según el modelo",
        xaxis_title="Fecha",
        yaxis_title="0 = Efectivo | 1 = Invertido",
        height=250,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], ticktext=["Efectivo (0)", "Invertido (1)"])
    return fig


def make_monthly_invested_bar(monthly: pd.DataFrame) -> go.Figure:
    if monthly is None or monthly.empty:
        return go.Figure()

    dfb = monthly.copy()
    dfb = dfb.reset_index()
    dfb["month"] = dfb["date"].dt.strftime("%Y-%m")

    fig = px.bar(
        dfb,
        x="month",
        y="pct_invertido",
        title="% del tiempo INVERTIDO por mes (ML)",
        labels={"month": "Mes", "pct_invertido": "% tiempo invertido"},
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(range=[0, 100])
    return fig


# -----------------------------
# UI: Header
# -----------------------------
st.markdown(
    """
<div class="dfs-hero">
  <div>
    <span class="dfs-badge">Piloto educativo</span>
    <span class="dfs-badge">Perfil de usuario</span>
    <span class="dfs-badge">Backtesting + ML</span>
  </div>
  <h1>DataFinscope – Piloto</h1>
  <p>
    Esta app es un <b>prototipo</b> para demostrar (con datos reales) dos cosas:
  </p>
  <ul style="margin-top:10px; margin-bottom:10px; color:#334155;">
    <li><b>Segmentación del usuario:</b> con 5 respuestas rápidas se clasifica el perfil (Conservador / Balanceado / Agresivo).</li>
    <li><b>Simulación histórica con ML:</b> se compara “comprar y mantener” vs. una estrategia donde un modelo estima <i>probabilidad de subida</i> y decide estar <b>invertido</b> o <b>en efectivo</b>.</li>
  </ul>
  <p><b>Importante:</b> no es recomendación financiera. Sirve para educación, evidencia de ciencia de datos y validación del prototipo.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


tabs = st.tabs(["🧠 Perfil del usuario", "📈 Backtesting + ML", "📚 Glosario"])


# ============================================================
# TAB 1: Perfil
# ============================================================
with tabs[0]:
    # Estado inicial (para que NO muestre perfil sin responder)
    if "perfil_calculado" not in st.session_state:
        st.session_state.perfil_calculado = False

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.subheader("Responde 5 preguntas (rápido)")
        st.caption("Esto ajusta el nivel de exposición recomendado en la simulación (threshold).")

        with st.container():
            c1, c2 = st.columns(2, gap="large")

            with c1:
                q_manejo = st.slider(
                    "1) ¿Qué tan cómodo/a te sientes usando apps o herramientas digitales? (1–5)",
                    1, 5, 3,
                    help="1 = me enredo mucho | 5 = me muevo fácil con apps, menús, plataformas.",
                )
                q_confianza = st.slider(
                    "2) ¿Qué tanta confianza te generan las plataformas/app de inversión? (1–5)",
                    1, 5, 3,
                    help="1 = desconfío totalmente | 5 = confío bastante (seguridad, reputación, claridad).",
                )

            with c2:
                q_utilidad = st.slider(
                    "3) ¿Qué tan útil te parece la información de inversión para tomar decisiones? (1–5)",
                    1, 5, 3,
                    help="1 = no me sirve | 5 = sí me ayuda a decidir con criterio.",
                )
                q_simulador = st.slider(
                    "4) ¿Qué tanto te interesa practicar con simulador/backtesting? (0–10)",
                    0, 10, 6,
                    help="0 = nada | 10 = me encanta practicar antes de invertir dinero real.",
                )

        q_uso = st.slider(
            "5) Si esto existiera hoy, ¿qué probabilidad hay de que lo uses en los próximos 3 meses? (1–10)",
            1, 10, 7,
            help="1 = nada probable | 10 = muy probable (lo usaría pronto).",
        )

        pago = st.number_input(
            "Opcional: ¿Cuánto pagarías al mes por una plataforma así? (COP)",
            min_value=0, value=15000, step=5000,
            help="Solo para tener una señal de disposición de pago (WTP). No afecta la simulación.",
        )

        colb1, colb2 = st.columns([0.35, 0.65])
        with colb1:
            calc = st.button("Calcular mi perfil", use_container_width=True)
        with colb2:
            st.caption("💡 Tip: no hay respuesta perfecta. Queremos entender tu estilo, no “pasar un examen”.")

        if calc:
            st.session_state.perfil_calculado = True

            # Scoring simple (0–100)
            score_conocimiento = ((q_manejo + q_utilidad + q_confianza) / 15) * 100
            score_riesgo = ((q_simulador + q_uso) / 20) * 100
            score_mixto = 0.55 * score_conocimiento + 0.45 * score_riesgo

            # Regla de perfil (puedes ajustar si quieres)
            if score_mixto >= 70:
                perfil = "Agresivo"
                threshold_rec = 0.50
                rec_text = "Mayor exposición al mercado (entra más fácil). Puede tener más variación en el camino."
            elif score_mixto >= 45:
                perfil = "Balanceado"
                threshold_rec = 0.55
                rec_text = "Equilibrio entre exposición y control de caídas. Buen punto medio para aprender."
            else:
                perfil = "Conservador"
                threshold_rec = 0.60
                rec_text = "Más filtro para entrar (menos exposición, menos “sustos”)."

            # Guardar en sesión
            st.session_state.perfil = perfil
            st.session_state.score_conocimiento = float(score_conocimiento)
            st.session_state.score_riesgo = float(score_riesgo)
            st.session_state.score_mixto = float(score_mixto)
            st.session_state.threshold_rec = float(threshold_rec)
            st.session_state.pago = int(pago)
            st.session_state.rec_text = rec_text

        # Mostrar resultados del perfil SOLO si ya se calculó
        if st.session_state.perfil_calculado:
            st.write("")
            sc = st.session_state.score_conocimiento
            sr = st.session_state.score_riesgo
            sm = st.session_state.score_mixto

            st.markdown(
                f"""
<div class="dfs-center-row">
  <div class="dfs-card" style="width: 280px;">
    <h4>Conocimiento</h4>
    <div class="big">{sc:.1f}/100</div>
    <div class="sub">Comodidad digital + utilidad de la info + confianza.</div>
  </div>
  <div class="dfs-card" style="width: 280px;">
    <h4>Riesgo / Interés</h4>
    <div class="big">{sr:.1f}/100</div>
    <div class="sub">Interés en simulación + intención de uso.</div>
  </div>
  <div class="dfs-card" style="width: 280px;">
    <h4>Mixto</h4>
    <div class="big">{sm:.1f}/100</div>
    <div class="sub">Resumen para recomendar nivel de exposición.</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.success(f"Perfil final: **{st.session_state.perfil}**")
            st.info(
                f"Recomendación para la simulación: **threshold {st.session_state.threshold_rec:.2f}**. {st.session_state.rec_text}"
            )

            with st.expander("📌 ¿Qué significa este resultado? (explicación clara)"):
                st.markdown(
                    """
**¿Qué es “perfil”?**  
Es una manera simple de resumir cómo te llevas con:  
1) herramientas digitales, 2) información para decidir, y 3) cuánto toleras variación.

**¿Cómo lo usamos en la simulación?**  
Ajustamos el **threshold del modelo**:

- **Threshold más alto** = el modelo exige más “confianza” para entrar → suele estar **menos tiempo invertido**.
- **Threshold más bajo** = entra con más facilidad → suele estar **más tiempo invertido**, pero con más cambios.

> Piensa en el threshold como un “filtro de seguridad” (más alto = más filtro).
"""
                )

    with right:
        st.subheader("Configuración recomendada (se usa en Backtesting + ML)")
        st.caption("Esto se llenará automáticamente cuando calcules tu perfil.")

        if st.session_state.perfil_calculado:
            st.markdown("**Sugerencia de threshold para tu perfil:**")
            st.metric("Threshold recomendado", f"{st.session_state.threshold_rec:.2f}")
            st.markdown("**Tu disposición de pago (opcional):**")
            st.metric("Pago mensual esperado (COP)", format_currency_cop(st.session_state.pago))
            st.markdown("---")
            st.markdown("Cuando estés listo, ve a la pestaña **📈 Backtesting + ML**.")
        else:
            st.info("Aún no has calculado el perfil. Responde y presiona **Calcular mi perfil**.")


# ============================================================
# TAB 2: Backtesting + ML
# ============================================================
with tabs[1]:
    # Cargar modelo
    try:
        model = joblib.load("modelo_ml_mercado.pkl")
    except Exception as e:
        st.error("No pude cargar `modelo_ml_mercado.pkl`. Verifica que esté en el repo junto a app.py.")
        st.stop()

    # Layout: izquierda perfil (solo resumen), derecha config inversión
    colL, colR = st.columns([1.05, 0.95], gap="large")

    with colL:
        st.subheader("1) Perfil (resumen)")
        if st.session_state.get("perfil_calculado", False):
            st.write(f"Perfil: **{st.session_state.perfil}**")
            st.write(f"Threshold recomendado: **{st.session_state.threshold_rec:.2f}**")
            st.caption("Puedes mover el threshold manualmente para ver el efecto (más alto = más conservador).")
        else:
            st.info("Tip: calcula tu perfil en la pestaña 🧠 para autocompletar el threshold sugerido.")

    with colR:
        st.subheader("2) Configuración de inversión (simulación)")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            ticker = st.text_input("Ticker (ej: AAPL, MSFT, TSLA)", value="AAPL")
            start_date = st.date_input("Fecha inicio", value=date(2021, 1, 1))

        with c2:
            capital_inicial = st.number_input("Capital inicial (USD)", min_value=100.0, value=1000.0, step=100.0)
            trm = st.number_input("TRM (COP por USD)", min_value=1000.0, value=4000.0, step=50.0)

        default_thr = float(st.session_state.get("threshold_rec", 0.55))
        threshold = st.slider(
            "Threshold ML (más alto = más conservador)",
            min_value=0.45, max_value=0.70, value=round(default_thr, 2), step=0.01,
            help="El modelo estima p_up (probabilidad de subir). Si p_up >= threshold → “invertido”.",
        )

        run_btn = st.button("Ejecutar backtesting ML", use_container_width=True)

    # No mostrar “contenedores vacíos” si aún no ejecuta
    if not run_btn:
        st.markdown("---")
        st.caption("💡 Ejecuta el backtesting para ver gráficas, métricas y tabla.")
    else:
        with st.spinner("Corriendo simulación… (descargando datos + calculando indicadores + backtesting)"):
            result = run_ml_backtest(
                model=model,
                ticker=ticker.strip().upper(),
                user_start_date=start_date,
                capital_inicial_usd=float(capital_inicial),
                trm_cop_usd=float(trm),
                threshold=float(threshold),
            )

        if "error" in result:
            st.error(result["error"])
            st.stop()

        df_res = result["df"]
        m = result["metrics"]
        figs = result["figs"]

        # -------------------------
        # 3) Resultados KPI centrados
        # -------------------------
        st.markdown("---")
        st.subheader("3) Resultados")

        # Cards con mejor look
        cap_m = m["capital_market_usd"]
        cap_s = m["capital_ml_usd"]
        sh_m = m["sh_market"]
        sh_s = m["sh_ml"]
        dd_m = m["mdd_market"]
        dd_s = m["mdd_ml"]

        # Row centrada con 4 tarjetas
        st.markdown(
            f"""
<div class="dfs-center-row">
  <div class="dfs-card" style="width: 290px;">
    <h4>Capital final (Market)</h4>
    <div class="big">{format_currency_usd(cap_m)}</div>
    <div class="sub">Comprar y mantener desde la fecha de inicio.</div>
  </div>
  <div class="dfs-card" style="width: 290px;">
    <h4>Capital final (Estrategia ML)</h4>
    <div class="big">{format_currency_usd(cap_s)}</div>
    <div class="sub">Entra/sale según la señal del modelo.</div>
  </div>
  <div class="dfs-card" style="width: 290px;">
    <h4>Sharpe (Market / ML)</h4>
    <div class="big">{sh_m:.2f} / {sh_s:.2f}</div>
    <div class="sub">Retorno promedio por “unidad de variación”.</div>
  </div>
  <div class="dfs-card" style="width: 290px;">
    <h4>Max Drawdown (Market / ML)</h4>
    <div class="big">{dd_m*100:.1f}% / {dd_s*100:.1f}%</div>
    <div class="sub">Peor caída desde un máximo (más cerca de 0 es mejor).</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ ¿Cómo interpretar Sharpe y Max Drawdown? (explicación detallada)"):
            st.markdown(
                """
### Sharpe (intuitivo)
- Piensa en Sharpe como: **“¿Cuánto gano por cada susto?”**
- **Más alto suele ser mejor**: significa que el retorno “compensa” mejor la variación.
- Si tu Sharpe es bajo, puede ser que el resultado se logró con mucha volatilidad.

### Max Drawdown (intuitivo)
- Drawdown es la **caída desde el punto más alto** que tu capital había alcanzado.
- Ejemplo: subes de 1000 a 1400 y luego bajas a 1200 → drawdown = (1200/1400 - 1) = **-14.3%**.
- No es “pérdida final”: es el **peor bajón durante el camino**.

### ¿Por qué mostramos Market / ML?
Porque queremos comparar:
- **Market (Buy & Hold)**: “me quedo invertido siempre”.
- **ML**: “entro y salgo buscando controlar caídas (pero puedo quedarme por fuera en subidas)”.
"""
            )

        # Zona de riesgo (dinámica)
        zone_label, zone_level = risk_zone_from_mdd(dd_s)
        msg = f"Zona de riesgo (ML) según Max Drawdown: **{dd_s*100:.1f}% → {zone_label}**"
        if zone_level == "success":
            st.success(msg)
        elif zone_level == "warning":
            st.warning(msg)
        elif zone_level == "error":
            st.error(msg)
        else:
            st.info(msg)

        with st.expander("🧭 ¿Qué significa “zona de riesgo”?"):
            st.markdown(
                """
La zona se basa en el **Max Drawdown** (peor caída histórica en el periodo simulado):

- **VERDE (bajo)**: caídas típicamente menores o iguales a ~10%.
- **AMARILLA (moderado)**: caídas entre ~10% y ~20% (hay “sustos”, pero manejables).
- **ROJA (alto)**: caídas mayores a ~20% (requiere tolerancia emocional/financiera).

> Esto NO es un diagnóstico definitivo del activo; solo describe lo que ocurrió en el periodo simulado.
"""
            )

        # -------------------------
        # 4) Gráficas “tipo trading”
        # -------------------------
        st.markdown("---")
        st.subheader("4) Gráficas (fáciles de leer)")

        st.plotly_chart(figs["candles"], use_container_width=True)
        with st.expander("📌 ¿Qué estoy viendo aquí? (velas + sombreado)"):
            st.markdown(
                """
- **Velas**: muestran el movimiento del precio (open/high/low/close).
- **Sombreado verde**: periodos donde el modelo estuvo **INVERTIDO** (pos_ml = 1).
- Si no hay sombreado en un tramo, el modelo estuvo en **EFECTIVO** (pos_ml = 0).

Esto ayuda a ver **en qué momentos** el modelo decidió entrar/salir, como si fuera un “overlay” encima del precio.
"""
            )

        st.plotly_chart(figs["capital"], use_container_width=True)
        with st.expander("📌 ¿Cómo leer “Capital acumulado”?"):
            st.markdown(
                """
- Esta gráfica muestra cómo habría evolucionado tu dinero con el tiempo.
- **Market (Buy & Hold)**: compras el activo al inicio y lo sostienes.
- **Estrategia (ML)**: solo participa cuando el modelo decide estar invertido.

**Lectura rápida:**
- Si la línea ML está arriba → el modelo “aportó valor” en ese periodo.
- Si está abajo → comprar y mantener fue mejor (pasa y es normal).
"""
            )

        st.plotly_chart(figs["drawdown"], use_container_width=True)
        with st.expander("📌 ¿Cómo leer “Drawdown”?"):
            st.markdown(
                """
- Drawdown mide “qué tan lejos estás de tu mejor momento” (máximo anterior).
- Mientras más cerca de **0**, mejor (menos bajones).
- Si una estrategia tiene drawdown menor, suele ser **más llevadera** para principiantes.

**Market vs ML**: compara cuál estrategia tuvo el “peor susto” más grande.
"""
            )

        # Estado + barra mensual
        cA, cB = st.columns([0.58, 0.42], gap="large")
        with cA:
            st.plotly_chart(figs["state"], use_container_width=True)
            with st.expander("📌 ¿Qué significa “Estado del inversionista”?"):
                st.markdown(
                    """
Esto es la decisión diaria del modelo (aplicada con shift para no hacer trampa):

- **1 = Invertido**: la estrategia participa en el mercado ese día.
- **0 = Efectivo**: la estrategia se sale (no asume el movimiento del precio).

No es “adivinar el futuro”: el modelo estima una probabilidad (**p_up**) y decide con el threshold.
"""
                )
        with cB:
            st.plotly_chart(figs["monthly"], use_container_width=True)
            with st.expander("📌 ¿Para qué sirve el % tiempo invertido por mes?"):
                st.markdown(
                    """
Esta barra responde una pregunta clave para principiantes:

**“¿Cuánto tiempo estuve realmente en el mercado?”**

- Si está cerca de 100%: el modelo estuvo casi siempre invertido (más exposición).
- Si está cerca de 0%: estuvo mayormente en efectivo (más conservador).
- Es una forma muy intuitiva de ver el “estilo” real del modelo según el threshold.
"""
                )

        # -------------------------
        # 5) Tabla final + explicación
        # -------------------------
        st.markdown("---")
        st.subheader("5) Tabla (últimos registros)")

        show_cols = [
            "close", "p_up", "signal_ml", "capital_market_usd", "capital_ml_usd",
            "capital_market_cop", "capital_ml_cop"
        ]

        df_view = df_res[show_cols].copy()
        df_view = df_view.rename(columns={
            "close": "Precio (USD)",
            "p_up": "Prob. de subida (p_up)",
            "signal_ml": "Decisión (0/1)",
            "capital_market_usd": "Capital Market (USD)",
            "capital_ml_usd": "Capital ML (USD)",
            "capital_market_cop": "Capital Market (COP)",
            "capital_ml_cop": "Capital ML (COP)",
        })

        # Formateo
        df_view["Precio (USD)"] = df_view["Precio (USD)"].round(2)
        df_view["Prob. de subida (p_up)"] = df_view["Prob. de subida (p_up)"].round(4)

        for c in ["Capital Market (USD)", "Capital ML (USD)"]:
            df_view[c] = df_view[c].round(2)

        for c in ["Capital Market (COP)", "Capital ML (COP)"]:
            df_view[c] = df_view[c].round(0).astype("int64")

        st.dataframe(df_view.tail(12), use_container_width=True)

        with st.expander("📌 ¿Qué significa cada columna de la tabla? (muy claro)"):
            st.markdown(
                """
Esta tabla es el “log” final del backtesting. Sirve para auditar qué hizo el modelo y cómo eso impactó tu capital.

- **Precio (USD)**: precio de cierre del día (en dólares).
- **Prob. de subida (p_up)**: estimación del modelo (0 a 1). Ej: 0.56 = 56% de probabilidad (según el modelo) de que el precio suba.
- **Decisión (0/1)**:
  - **1** → el modelo dice “me quedo invertido”.
  - **0** → el modelo dice “me salgo a efectivo”.
- **Capital Market (USD/COP)**: cuánto valdría tu capital si hubieras hecho Buy & Hold.
- **Capital ML (USD/COP)**: cuánto valdría tu capital siguiendo la estrategia ML.

**Objetivo de esta tabla**: que cualquier persona (o jurado) pueda ver “qué decidió el modelo” y “qué efecto tuvo”.
"""
            )


# ============================================================
# TAB 3: Glosario
# ============================================================
with tabs[2]:
    st.subheader("Glosario rápido (para usuarios sin experiencia)")
    st.markdown(
        """
- **Ticker**: código del activo en bolsa (ej: AAPL = Apple).
- **Backtesting**: probar una estrategia en datos históricos para ver cómo habría funcionado.
- **p_up**: probabilidad estimada por el modelo de que el precio suba (según sus variables).
- **Threshold**: “filtro” para entrar. Si p_up ≥ threshold → invertido.
- **Sharpe**: retorno por unidad de variación (“qué tanto gano por cada susto”).
- **Drawdown**: caída desde el máximo. Es el peor bajón durante el camino.
- **Buy & Hold**: comprar y mantener sin hacer nada.
- **TRM**: tasa de cambio (COP por USD), para entender el capital también en pesos.
"""
    )

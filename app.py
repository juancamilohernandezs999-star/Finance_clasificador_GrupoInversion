import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go

# -----------------------
# Page config + Style
# -----------------------
st.set_page_config(page_title="DataFinscope – Piloto", layout="wide")

st.markdown(
    """
    <style>
      .dfs-hero {
        padding: 18px 22px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(40,80,240,0.12), rgba(20,200,160,0.10));
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 14px;
      }
      .dfs-hero h1 { margin: 0; font-size: 34px; }
      .dfs-hero p { margin: 6px 0 0; color: rgba(0,0,0,0.70); }
      .dfs-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(0,0,0,0.06);
        margin-right: 8px;
        font-size: 12px;
      }
      .dfs-section-title {
        margin-top: 6px;
        margin-bottom: 4px;
        font-weight: 700;
        font-size: 18px;
      }
      .dfs-note {
        color: rgba(0,0,0,0.65);
        font-size: 13px;
      }
      .dfs-divider {
        height: 1px;
        background: rgba(0,0,0,0.08);
        margin: 14px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dfs-hero">
      <span class="dfs-pill">Piloto educativo</span>
      <span class="dfs-pill">Perfil de usuario</span>
      <span class="dfs-pill">Backtesting + ML</span>
      <h1>DataFinscope – Piloto</h1>
      <p>
        Segmentación de usuario + simulación histórica con Machine Learning.
        <b>No es recomendación financiera</b>.
      </p>
      <p class="dfs-note">
        Objetivo: que un usuario sin experiencia entienda su perfil y vea cómo se comportaría una estrategia
        (versus comprar y mantener) usando datos reales.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helpers: métricas
# -----------------------
def sharpe_ratio(returns, rf=0.0):
    returns = pd.Series(returns).dropna()
    if len(returns) < 10:
        return np.nan
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (returns.mean() - rf) / std * np.sqrt(252)

def max_drawdown(cum_curve):
    cum_curve = pd.Series(cum_curve).dropna()
    if cum_curve.empty:
        return np.nan, pd.Series(dtype=float)
    peak = cum_curve.cummax()
    dd = (cum_curve / peak) - 1
    return float(dd.min()), dd

def perfil_por_score(score_mixto):
    if score_mixto < 35:
        return "Conservador"
    elif score_mixto < 70:
        return "Balanceado"
    else:
        return "Agresivo"

def recomendacion_por_perfil(perfil):
    if perfil == "Conservador":
        return {"estrategia": "ML", "threshold": 0.60, "mensaje": "Entra menos veces al mercado: prioriza estabilidad."}
    if perfil == "Balanceado":
        return {"estrategia": "ML", "threshold": 0.55, "mensaje": "Equilibrio entre retorno y control de caídas."}
    return {"estrategia": "ML", "threshold": 0.50, "mensaje": "Mayor exposición: más movimiento (para bien o mal)."}

def fmt_usd(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def fmt_cop(x):
    try:
        return f"${x:,.0f} COP"
    except Exception:
        return str(x)

# -----------------------
# Data + Features (mercado)
# -----------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str):
    df = yf.download(ticker, start=start, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        try:
            price = df[("Close", ticker)]
        except Exception:
            try:
                price = df.xs("Close", axis=1, level=0).iloc[:, 0]
            except Exception:
                return pd.DataFrame()

        try:
            volume = df[("Volume", ticker)]
        except Exception:
            try:
                volume = df.xs("Volume", axis=1, level=0).iloc[:, 0]
            except Exception:
                volume = pd.Series(index=df.index, data=np.nan)

        out = pd.DataFrame({"price": price, "volume": volume})
    else:
        price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if price_col is None or "Volume" not in df.columns:
            return pd.DataFrame()
        out = df[[price_col, "Volume"]].rename(columns={price_col: "price", "Volume": "volume"})

    out = out.dropna(subset=["price"])
    out["ret"] = out["price"].pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def make_market_features(df, ma_fast=20, ma_slow=50, vol_win=20):
    full = df.copy()
    full["ma20"] = full["price"].rolling(ma_fast).mean()
    full["ma50"] = full["price"].rolling(ma_slow).mean()

    full["dist_ma20"] = (full["price"] / full["ma20"]) - 1
    full["dist_ma50"] = (full["price"] / full["ma50"]) - 1

    full["ret_1"] = full["ret"].shift(1)
    full["ret_5"] = full["price"].pct_change(5)
    full["ret_10"] = full["price"].pct_change(10)
    full["vol20"] = full["ret"].rolling(vol_win).std()
    full["mom20"] = full["price"] / full["price"].shift(20) - 1

    features = ["dist_ma20", "dist_ma50", "ret_1", "ret_5", "ret_10", "vol20", "mom20"]
    full = full.replace([np.inf, -np.inf], np.nan)
    full = full.dropna(subset=features + ["ret"])
    X = full[features].astype(float)
    return X, full, features

def apply_plot_style(fig, height=420):
    fig.update_layout(
        height=height,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # “TradingView vibes”: rangeslider
    fig.update_xaxes(rangeslider=dict(visible=True))
    return fig

def plot_capital(dates, market, strategy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=market, mode="lines", name="Market (Buy & Hold)"))
    fig.add_trace(go.Scatter(x=dates, y=strategy, mode="lines", name="Estrategia (ML)"))
    fig.update_layout(title="Capital acumulado (USD)", xaxis_title="Fecha", yaxis_title="Capital (USD)")
    return apply_plot_style(fig, height=420)

def plot_drawdown(dates, dd_mkt, dd_str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd_mkt, mode="lines", name="Drawdown Market"))
    fig.add_trace(go.Scatter(x=dates, y=dd_str, mode="lines", name="Drawdown ML"))
    fig.update_layout(title="Drawdown (caídas desde el máximo)", xaxis_title="Fecha", yaxis_title="Drawdown")
    return apply_plot_style(fig, height=320)

def plot_state(dates, signal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=signal, mode="lines", fill="tozeroy", name="Estado"))
    fig.update_layout(
        title="Estado del inversionista (0 = Efectivo, 1 = Invertido)",
        xaxis_title="Fecha",
        yaxis=dict(tickvals=[0, 1], ticktext=["Efectivo", "Invertido"]),
    )
    return apply_plot_style(fig, height=260)

# -----------------------
# Load model
# -----------------------
@st.cache_resource(show_spinner=False)
def load_market_model():
    return joblib.load("modelo_ml_mercado.pkl")

model = load_market_model()

# -----------------------
# Session state
# -----------------------
if "perfil" not in st.session_state:
    st.session_state.perfil = None
if "rec" not in st.session_state:
    st.session_state.rec = None

# -----------------------
# Sidebar inputs (config)
# -----------------------
st.sidebar.markdown("### Configuración")
ticker = st.sidebar.text_input("Ticker (ej: AAPL, MSFT, TSLA)", value="AAPL")
start = st.sidebar.date_input("Fecha inicio", value=pd.to_datetime("2021-01-01"))
capital = st.sidebar.number_input("Capital inicial (USD)", min_value=100, value=1000, step=100)
trm = st.sidebar.number_input("TRM (COP por USD)", min_value=1000, value=4000, step=50)

st.sidebar.markdown("<div class='dfs-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Nota")
st.sidebar.caption("Usa al menos 6–12 meses de historia para calcular MA50, momentum, etc.")

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["🧭 Perfil", "📈 Simulación ML", "📚 Glosario"])

# -----------------------
# TAB 1: Perfil
# -----------------------
with tab1:
    st.markdown("<div class='dfs-section-title'>Perfil del usuario</div>", unsafe_allow_html=True)
    st.caption("Responde rápido. Esto ajusta el nivel de exposición recomendado en la simulación.")

    with st.expander("Responder (5 preguntas rápidas)", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            manejo_digital = st.slider(
                "Comodidad usando apps (1–5)",
                1, 5, 3,
                help="1 = me cuesta mucho, 5 = manejo apps sin problema (bancos, pagos, inversión, etc.)"
            )
            confianza_apps = st.slider(
                "Confianza en plataformas (1–5)",
                1, 5, 3,
                help="1 = no confío, 5 = confío bastante en plataformas reguladas/conocidas."
            )

        with col2:
            utilidad_info = st.slider(
                "La info me ayuda a decidir (1–5)",
                1, 5, 3,
                help="1 = no me sirve, 5 = sí me ayuda a entender y decidir mejor."
            )
            prob_backtesting = st.slider(
                "Interés en simulador histórico (0–10)",
                0, 10, 7,
                help="0 = nada, 10 = lo usaría muchísimo para aprender sin arriesgar."
            )

        with col3:
            prob_uso = st.slider(
                "Probabilidad de usar la plataforma (1–10)",
                1, 10, 7,
                help="1 = poco probable, 10 = muy probable en los próximos meses."
            )
            pago_mensual = st.number_input(
                "Pago mensual esperado (COP)",
                min_value=0, value=0, step=5000,
                help="No es cobro real. Solo ayuda a entender intención/compromiso."
            )

        if st.button("Calcular mi perfil", type="primary"):
            score_con = (manejo_digital/5 + utilidad_info/5 + confianza_apps/5) / 3 * 100
            score_rie = ((prob_uso/10) + (prob_backtesting/10)) / 2 * 100
            score_mix = 0.5 * score_con + 0.5 * score_rie

            perfil = perfil_por_score(score_mix)
            rec = recomendacion_por_perfil(perfil)

            st.session_state.perfil = {
                "score_conocimiento": score_con,
                "score_riesgo": score_rie,
                "score_mixto": score_mix,
                "perfil_final": perfil,
                "pago_mensual_cop": pago_mensual
            }
            st.session_state.rec = rec

    if st.session_state.perfil:
        p = st.session_state.perfil
        rec = st.session_state.rec

        a, b, c = st.columns(3)
        a.metric("Score conocimiento", f"{p['score_conocimiento']:.1f}/100",
                 help="Promedio de comodidad usando apps + utilidad de info + confianza.")
        b.metric("Score riesgo/interés", f"{p['score_riesgo']:.1f}/100",
                 help="Promedio de intención de uso + interés en simulador.")
        c.metric("Score mixto", f"{p['score_mixto']:.1f}/100",
                 help="Promedio 50/50: conocimiento y riesgo/interés.")

        st.success(f"Perfil final: **{p['perfil_final']}**")
        st.info(f"Recomendación: **{rec['estrategia']}** con threshold **{rec['threshold']}**. {rec['mensaje']}")

        with st.expander("¿Qué significa mi perfil? (explicación simple)"):
            st.markdown(
                f"""
                **Tu perfil: {p['perfil_final']}**  
                - **Conservador:** prefiere menos sustos (menos tiempo invertido).  
                - **Balanceado:** busca equilibrio (no quedarse por fuera, pero tampoco entrar a ciegas).  
                - **Agresivo:** tolera variación y busca mayor exposición.  

                **Cómo lo usamos en la simulación:** ajustamos el *threshold* del modelo.  
                - Threshold **más alto** = el modelo exige más “seguridad” para entrar (más conservador).  
                - Threshold **más bajo** = entra más fácil (más agresivo).
                """
            )

# -----------------------
# TAB 2: Simulación ML
# -----------------------
with tab2:
    st.markdown("<div class='dfs-section-title'>Simulación (Machine Learning de mercado)</div>", unsafe_allow_html=True)
    st.caption("Compara: Buy & Hold vs Estrategia ML. Datos reales descargados con yfinance.")

    default_threshold = 0.55
    if st.session_state.rec:
        default_threshold = float(st.session_state.rec["threshold"])

    threshold = st.slider(
        "Threshold ML (más alto = más conservador)",
        0.50, 0.70, float(default_threshold), 0.01,
        help="Probabilidad mínima (p_up) para decidir estar invertido. Ej: 0.60 = más exigente."
    )

    run = st.button("Ejecutar backtesting ML", type="primary")

    if run:
        with st.spinner("Cargando precios y ejecutando backtesting..."):
            df = load_prices(ticker, str(start))

            if df.empty:
                st.error("No se pudo descargar data. Revisa el ticker o intenta más tarde.")
                st.stop()

            if len(df) < 120:
                st.warning(
                    f"Hay pocos datos desde esa fecha ({len(df)} filas). "
                    "Elige una fecha de inicio más antigua (ideal 6–12 meses atrás)."
                )
                st.stop()

            X, full, features = make_market_features(df)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()

            if X.empty or len(X) < 5:
                st.warning("No quedaron filas suficientes después de calcular indicadores. Prueba otra fecha.")
                st.stop()

            proba = model.predict_proba(X.astype(float))[:, 1]

            full = full.loc[X.index].copy()
            full["p_up"] = proba
            full["signal_ml"] = (full["p_up"] >= threshold).astype(int)

            # Market
            full["cum_market"] = (1 + full["ret"]).cumprod()
            full["capital_market_usd"] = capital * full["cum_market"]
            full["capital_market_cop"] = full["capital_market_usd"] * trm

            # ML
            full["strategy_ret_ml"] = full["signal_ml"].shift(1) * full["ret"]
            full["cum_strategy_ml"] = (1 + full["strategy_ret_ml"]).cumprod()
            full["capital_strategy_ml_usd"] = capital * full["cum_strategy_ml"]
            full["capital_strategy_ml_cop"] = full["capital_strategy_ml_usd"] * trm

            sh_mkt = sharpe_ratio(full["ret"])
            sh_ml = sharpe_ratio(full["strategy_ret_ml"])

            mdd_mkt, dd_mkt = max_drawdown(full["cum_market"])
            mdd_ml, dd_ml = max_drawdown(full["cum_strategy_ml"])

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Capital final (Market USD)", fmt_usd(full["capital_market_usd"].iloc[-1]),
                  help="Resultado si compras y mantienes (Buy & Hold).")
        k2.metric("Capital final (ML USD)", fmt_usd(full["capital_strategy_ml_usd"].iloc[-1]),
                  help="Resultado usando el modelo ML para entrar/salir.")
        k3.metric("Sharpe (Market / ML)", f"{sh_mkt:.2f} / {sh_ml:.2f}",
                  help="Retorno ajustado por riesgo. >1 suele considerarse bueno.")
        k4.metric("Max Drawdown (Market / ML)", f"{mdd_mkt:.1%} / {mdd_ml:.1%}",
                  help="Peor caída desde el máximo. Ej: -0.20 = -20%.")

        # Semáforo
        dd_abs = abs(mdd_ml) if not np.isnan(mdd_ml) else np.nan
        if np.isnan(dd_abs):
            st.info("No se pudo calcular drawdown (datos insuficientes o retornos nulos).")
        elif dd_abs <= 0.10:
            st.success(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona VERDE (bajo)")
        elif dd_abs <= 0.25:
            st.warning(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona AMARILLA (moderado)")
        else:
            st.error(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona ROJA (alto)")

        # Charts + Explain expanders
        st.plotly_chart(plot_capital(full.index, full["capital_market_usd"], full["capital_strategy_ml_usd"]), use_container_width=True)
        with st.expander("📌 ¿Cómo leer 'Capital acumulado'?"):
            st.markdown(
                """
                **Qué muestra:** cómo evoluciona tu dinero con el tiempo.  
                - **Market (Buy & Hold):** compras el activo y no haces nada.  
                - **Estrategia (ML):** el modelo decide si estar invertido o en efectivo.

                **Interpretación rápida:**  
                - Si la línea ML está arriba → el modelo está aportando valor en ese periodo.  
                - Si está abajo → el mercado fue mejor (pasa y es normal).  
                """
            )

        st.plotly_chart(plot_drawdown(full.index, dd_mkt, dd_ml), use_container_width=True)
        with st.expander("📌 ¿Cómo leer 'Drawdown'? (sustos del camino)"):
            st.markdown(
                """
                **Drawdown = caída desde el máximo** (no es pérdida final).  
                Ejemplo: si vas en $2,000, subes a $2,400 y luego bajas a $2,100, tu drawdown es **-12.5%**.

                **Para qué sirve:**  
                - Te muestra la **peor caída** y la **volatilidad emocional** de la estrategia.  
                - Un drawdown menor suele ser más “llevadero” para principiantes.
                """
            )

        st.plotly_chart(plot_state(full.index, full["signal_ml"]), use_container_width=True)
        with st.expander("📌 ¿Qué significa 'Estado del inversionista'?"):
            st.markdown(
                """
                Esto es la decisión diaria del modelo:  
                - **1 = Invertido:** la estrategia está en el mercado.  
                - **0 = Efectivo:** la estrategia se sale para reducir riesgo.

                **Importante:**  
                El modelo no “adivina” el futuro. Solo estima probabilidad de subida (**p_up**) con variables técnicas
                y decide según el *threshold*.
                """
            )

        # Table
        st.markdown("<div class='dfs-section-title'>Tabla (últimos registros)</div>", unsafe_allow_html=True)
        table = full[["price", "p_up", "signal_ml", "capital_market_usd", "capital_strategy_ml_usd", "capital_market_cop", "capital_strategy_ml_cop"]].tail(30).copy()
        table = table.rename(columns={
            "price": "Precio (USD)",
            "p_up": "Prob. de subida (p_up)",
            "signal_ml": "Decisión (0/1)",
            "capital_market_usd": "Capital Market (USD)",
            "capital_strategy_ml_usd": "Capital ML (USD)",
            "capital_market_cop": "Capital Market (COP)",
            "capital_strategy_ml_cop": "Capital ML (COP)",
        })

        # redondeos
        for c in ["Precio (USD)", "Prob. de subida (p_up)"]:
            table[c] = table[c].astype(float).round(4)

        for c in ["Capital Market (USD)", "Capital ML (USD)", "Capital Market (COP)", "Capital ML (COP)"]:
            table[c] = table[c].astype(float).round(2)

        st.dataframe(table, use_container_width=True)

        with st.expander("📌 ¿Qué es p_up y por qué aparece 0/1?"):
            st.markdown(
                """
                - **p_up**: probabilidad (0 a 1) de que el precio suba (según el modelo).  
                - **Decisión (0/1)**:
                  - **1** si `p_up >= threshold` → “me quedo invertido”.
                  - **0** si `p_up < threshold` → “me salgo a efectivo”.

                El threshold actúa como “filtro de confianza”.
                """
            )

# -----------------------
# TAB 3: Glosario
# -----------------------
with tab3:
    st.markdown("<div class='dfs-section-title'>Glosario rápido (para no expertos)</div>", unsafe_allow_html=True)

    with st.expander("Sharpe"):
        st.write("Mide el retorno por unidad de riesgo. Valores > 1 suelen considerarse buenos (depende del contexto).")

    with st.expander("Drawdown"):
        st.write("La peor caída desde un máximo. Describe el ‘peor susto’ de una estrategia.")

    with st.expander("Buy & Hold"):
        st.write("Comprar y mantener sin operar. Sirve como benchmark (línea base).")

    with st.expander("Backtesting"):
        st.write("Probar una estrategia con datos históricos para ver cómo habría funcionado en el pasado.")

    with st.expander("p_up / threshold"):
        st.write("p_up es la probabilidad de subida estimada por el modelo. threshold es el mínimo para entrar al mercado.")

    with st.expander("Indicadores usados (features)"):
        st.markdown(
            """
            - **MA20/MA50:** promedio de precio en 20/50 días.  
            - **dist_ma20/dist_ma50:** qué tan lejos está el precio de esas medias.  
            - **ret_5/ret_10:** retorno a 5/10 días.  
            - **vol20:** volatilidad (variación) en 20 días.  
            - **mom20:** momentum 20 días (fuerza de tendencia).
            """
        )

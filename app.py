import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="DataFinscope – Piloto", layout="wide")

# -----------------------
# Helpers: métricas
# -----------------------
def sharpe_ratio(returns, rf=0.0):
    returns = pd.Series(returns).dropna()
    if returns.std() == 0 or len(returns) < 10:
        return np.nan
    return (returns.mean() - rf) / returns.std() * np.sqrt(252)

def max_drawdown(cum_curve):
    cum_curve = pd.Series(cum_curve).dropna()
    peak = cum_curve.cummax()
    dd = (cum_curve / peak) - 1
    return float(dd.min()), dd

def minmax_0_100(x):
    x = pd.Series(x).astype(float)
    return 100 * (x - x.min()) / (x.max() - x.min() + 1e-9)

def perfil_por_score(score_mixto):
    # Ajusta si quieres: cortes recomendados para MVP
    if score_mixto < 35:
        return "Conservador"
    elif score_mixto < 70:
        return "Balanceado"
    else:
        return "Agresivo"

def recomendacion_por_perfil(perfil):
    # Conecta tu ML de mercado (lo que ya cerraste)
    if perfil == "Conservador":
        return {"estrategia": "ML", "threshold": 0.60, "mensaje": "Prioriza estabilidad y minimizar caídas."}
    if perfil == "Balanceado":
        return {"estrategia": "ML", "threshold": 0.55, "mensaje": "Equilibrio entre retorno y riesgo."}
    return {"estrategia": "ML", "threshold": 0.50, "mensaje": "Mayor exposición al mercado, mayor variación."}

# -----------------------
# Data + Features (mercado)
# -----------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str):
    df = yf.download(ticker, start=start, progress=False)
    # yfinance a veces entrega columnas MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # toma el primer nivel "Close"/"Adj Close" si existe
        if ("Close", ticker) in df.columns:
            price = df[("Close", ticker)]
        else:
            # fallback
            price = df.xs("Close", axis=1, level=0).iloc[:, 0]
        if ("Volume", ticker) in df.columns:
            volume = df[("Volume", ticker)]
        else:
            volume = df.xs("Volume", axis=1, level=0).iloc[:, 0]
        out = pd.DataFrame({"price": price, "volume": volume})
    else:
        # si auto_adjust=True, a veces no viene "Adj Close"
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        out = df[[price_col, "Volume"]].rename(columns={price_col: "price", "Volume": "volume"})

    out = out.dropna()
    out["ret"] = out["price"].pct_change()
    return out.dropna()

def make_market_features(df, ma_fast=20, ma_slow=50, vol_win=20):
    X = df.copy()
    X["ma20"] = X["price"].rolling(ma_fast).mean()
    X["ma50"] = X["price"].rolling(ma_slow).mean()
    X["dist_ma20"] = (X["price"] / X["ma20"]) - 1
    X["dist_ma50"] = (X["price"] / X["ma50"]) - 1
    X["ret_1"] = X["ret"].shift(1)
    X["ret_5"] = X["price"].pct_change(5)
    X["ret_10"] = X["price"].pct_change(10)
    X["vol20"] = X["ret"].rolling(vol_win).std()
    X["mom20"] = X["price"] / X["price"].shift(20) - 1

    features = ["dist_ma20","dist_ma50","ret_1","ret_5","ret_10","vol20","mom20"]
    X = X.dropna(subset=features + ["ret"])
    return X[features], X

def plot_capital(dates, market, strategy, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=market, mode="lines", name="Market (Buy & Hold)"))
    fig.add_trace(go.Scatter(x=dates, y=strategy, mode="lines", name="Strategy (ML)"))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Capital (USD)", height=420)
    return fig

def plot_drawdown(dates, dd_mkt, dd_str, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd_mkt, mode="lines", name="Drawdown Market"))
    fig.add_trace(go.Scatter(x=dates, y=dd_str, mode="lines", name="Drawdown Strategy"))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Drawdown", height=320)
    return fig

def plot_state(dates, signal, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=signal, mode="lines", fill="tozeroy", name="Estado"))
    fig.update_layout(
        title=title, xaxis_title="Fecha",
        yaxis=dict(tickvals=[0,1], ticktext=["Efectivo","Invertido"]),
        height=260
    )
    return fig

# -----------------------
# UI
# -----------------------
st.title("DataFinscope – Piloto (Perfil + Simulación)")
st.caption("Piloto educativo: segmentación de usuario + backtesting histórico. No es recomendación financiera.")

# Cargar modelo de mercado (guardado por ti)
@st.cache_resource(show_spinner=False)
def load_market_model():
    return joblib.load("modelo_ml_mercado.pkl")

model = load_market_model()

# Estado para guardar perfil y recomendación
if "perfil" not in st.session_state:
    st.session_state.perfil = None
if "rec" not in st.session_state:
    st.session_state.rec = None

with st.expander("1) Perfil del usuario (desplegar y responder)", expanded=True):
    st.subheader("Cuéntanos 5 cosas (rápido)")

    col1, col2, col3 = st.columns(3)
    with col1:
        manejo_digital = st.slider("Manejo digital (1–5)", 1, 5, 3)
        confianza_apps = st.slider("Confianza en apps (1–5)", 1, 5, 3)
    with col2:
        utilidad_info = st.slider("Utilidad de info de inversión (1–5)", 1, 5, 3)
        prob_backtesting = st.slider("Probabilidad de usar simulador/backtesting (0–10)", 0, 10, 7)
    with col3:
        prob_uso = st.slider("Probabilidad de uso (1–10)", 1, 10, 7)
        # opcional, puede ser COP mensual como proxy de compromiso
        pago_mensual = st.number_input("¿Cuánto pagarías al mes? (COP)", min_value=0, value=0, step=5000)

    if st.button("Calcular mi perfil"):
        # Scores simples (MVP). Luego los ajustamos a tus columnas reales.
        # Conocimiento: manejo + utilidad + confianza (0-100)
        score_con = (manejo_digital/5 + utilidad_info/5 + confianza_apps/5)/3 * 100

        # Riesgo/interés: prob_uso + prob_backtesting (0-100)
        score_rie = ((prob_uso/10) + (prob_backtesting/10))/2 * 100

        # Mixto 50/50 (puedes ponderar 60/40 si quieres)
        score_mix = 0.5*score_con + 0.5*score_rie

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
        a.metric("Score conocimiento", f"{p['score_conocimiento']:.1f}/100")
        b.metric("Score riesgo/interés", f"{p['score_riesgo']:.1f}/100")
        c.metric("Score mixto", f"{p['score_mixto']:.1f}/100")

        st.success(f"Perfil final: **{p['perfil_final']}**")
        st.info(f"Recomendación: **{rec['estrategia']}** con threshold **{rec['threshold']}**. {rec['mensaje']}")

st.divider()

st.subheader("2) Simulación (Machine Learning de mercado)")
colA, colB, colC, colD = st.columns(4)
with colA:
    ticker = st.text_input("Ticker", value="AAPL")
with colB:
    start = st.date_input("Fecha inicio", value=pd.to_datetime("2021-01-01"))
with colC:
    capital = st.number_input("Capital inicial (USD)", min_value=100, value=1000, step=100)
with colD:
    trm = st.number_input("TRM (COP por USD)", min_value=1000, value=4000, step=50)

# Usa threshold recomendado si ya hay perfil
default_threshold = 0.55
if st.session_state.rec:
    default_threshold = st.session_state.rec["threshold"]

threshold = st.slider("Threshold ML (más alto = más conservador)", 0.50, 0.70, float(default_threshold), 0.01)

run = st.button("Ejecutar backtesting ML")

if run:
    with st.spinner("Cargando precios y ejecutando backtesting..."):
        df = load_prices(ticker, str(start))

        X, full = make_market_features(df)
        proba = model.predict_proba(X)[:, 1]

        full = full.loc[X.index].copy()
        full["p_up"] = proba
        full["signal_ml"] = (full["p_up"] >= threshold).astype(int)

        # market curve
        full["cum_market"] = (1 + full["ret"]).cumprod()
        full["capital_market_usd"] = capital * full["cum_market"]
        full["capital_market_cop"] = full["capital_market_usd"] * trm

        # ML strategy
        full["strategy_ret_ml"] = full["signal_ml"].shift(1) * full["ret"]
        full["cum_strategy_ml"] = (1 + full["strategy_ret_ml"]).cumprod()
        full["capital_strategy_ml_usd"] = capital * full["cum_strategy_ml"]
        full["capital_strategy_ml_cop"] = full["capital_strategy_ml_usd"] * trm

        sh_mkt = sharpe_ratio(full["ret"])
        sh_ml = sharpe_ratio(full["strategy_ret_ml"])

        mdd_mkt, dd_mkt = max_drawdown(full["cum_market"])
        mdd_ml, dd_ml = max_drawdown(full["cum_strategy_ml"])

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Capital final (Market USD)", f"${full['capital_market_usd'].iloc[-1]:,.0f}")
    k2.metric("Capital final (ML USD)", f"${full['capital_strategy_ml_usd'].iloc[-1]:,.0f}")
    k3.metric("Sharpe (Market / ML)", f"{sh_mkt:.2f} / {sh_ml:.2f}")
    k4.metric("Max Drawdown (Market / ML)", f"{mdd_mkt:.1%} / {mdd_ml:.1%}")

    # Semáforo (simple)
    dd_abs = abs(mdd_ml)
    if dd_abs <= 0.10:
        st.success(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona VERDE (bajo)")
    elif dd_abs <= 0.25:
        st.warning(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona AMARILLA (moderado)")
    else:
        st.error(f"Riesgo (Drawdown ML): {mdd_ml:.1%} → Zona ROJA (alto)")

    # Gráficas
    fig1 = plot_capital(full.index, full["capital_market_usd"], full["capital_strategy_ml_usd"], "Capital acumulado (USD)")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = plot_drawdown(full.index, dd_mkt, dd_ml, "Drawdown (Market vs ML)")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = plot_state(full.index, full["signal_ml"], "Estado del inversionista (0 = efectivo, 1 = invertido)")
    st.plotly_chart(fig3, use_container_width=True)

    # Tabla resumen corta
    st.subheader("Tabla (últimos registros)")
    show_cols = ["price","p_up","signal_ml","capital_market_usd","capital_strategy_ml_usd","capital_market_cop","capital_strategy_ml_cop"]
    st.dataframe(full[show_cols].tail(20))

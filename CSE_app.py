import streamlit as st
import pandas as pd
import datetime
import requests
import yfinance as yf
import duckduckgo_search as ddg
import docx
from io import BytesIO
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import numpy as np
import re

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CSE Stock Dashboard", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
:root { --primary-dark: #002B5B; --primary-medium: #256D85; --primary-light: #DFF6FF;
        --accent-blue: #4a6fa5; --accent-teal: #00ADB5; --text-light: #333333; --text-dark: #f0f2f6;
        --bg-light: #f8f9fa; --bg-dark: #0e1117; }
* { font-family: 'Lato', 'Segoe UI', Roboto, sans-serif; }
.stSidebar { background: linear-gradient(135deg, var(--primary-dark), var(--primary-medium)) !important; color:white; }
.footer { font-size: 0.78rem; text-align: center; margin-top: 3rem; color: var(--primary-medium); padding: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
@st.cache_data
def get_yf_data(ticker, start, end):
    cse_ticker = f"{ticker}.CM" if not ticker.endswith(".CM") else ticker
    stock = yf.Ticker(cse_ticker)
    return stock.history(start=start, end=end), stock.info, stock

def macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def zigzag_with_signals(df, threshold=0.03):
    df['ZigZag'] = np.nan
    last_pivot = df['Close'].iloc[0]
    pivots = []
    direction = None
    for i in range(1, len(df)):
        change = (df['Close'].iloc[i] - last_pivot) / last_pivot
        if (direction != 'up') and (change > threshold):
            direction = 'up'
            last_pivot = df['Close'].iloc[i]
            df['ZigZag'].iloc[i] = df['Close'].iloc[i]
            pivots.append((df.index[i], df['Close'].iloc[i], 'buy'))
        elif (direction != 'down') and (change < -threshold):
            direction = 'down'
            last_pivot = df['Close'].iloc[i]
            df['ZigZag'].iloc[i] = df['Close'].iloc[i]
            pivots.append((df.index[i], df['Close'].iloc[i], 'sell'))
    return df, pivots

def buy_sell_signal(macd_line, signal_line, rsi_val):
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_rsi = rsi_val.iloc[-1]
    if latest_macd > latest_signal and latest_rsi < 70:
        return "BUY", f"MACD bullish crossover & RSI {latest_rsi:.2f}"
    elif latest_macd < latest_signal and latest_rsi > 30:
        return "SELL", f"MACD bearish crossover & RSI {latest_rsi:.2f}"
    else:
        return "HOLD", f"No strong signal — RSI {latest_rsi:.2f}"

@st.cache_data
def fetch_duckduckgo_news(query):
    try:
        return ddg.ddg_news(query, max_results=5)
    except:
        return []

@st.cache_data
def scrape_economynext():
    try:
        res = requests.get("https://economynext.com/category/business/", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return [a.text for a in soup.find_all("a", href=True) if a.text.strip()][:10]
    except:
        return []

# ---------- SIDEBAR ----------
st.sidebar.header("CSE Input")
ticker = st.sidebar.text_input("CSE Ticker (e.g. WIND-N0000):", "WIND-N0000")
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
use_websearch = st.sidebar.checkbox("Enable Web Search", value=False)

# ---------- FETCH ----------
hist, info, stock_obj = get_yf_data(ticker, start_date, end_date)

if hist is not None and not hist.empty:
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    macd_line, signal_line = macd(hist)
    hist["RSI"] = rsi(hist)
    hist, pivots = zigzag_with_signals(hist)
    signal, reason = buy_sell_signal(macd_line, signal_line, hist["RSI"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Chart", "📈 Technical Indicators", "🧠 Recommendation", "📰 Market News"
    ])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
            name="Price", increasing_line_color='green', decreasing_line_color='red'
        ))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], mode='lines', name='SMA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA50"], mode='lines', name='SMA50', line=dict(color='orange')))
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='lightgrey', opacity=0.4, yaxis='y2'))

        for date, price, action in pivots:
            if action == 'buy':
                fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers+text', text=["Buy"],
                                         name='Buy Signal', marker=dict(color='green', size=12, symbol='triangle-up'),
                                         textposition="top center"))
            elif action == 'sell':
                fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers+text', text=["Sell"],
                                         name='Sell Signal', marker=dict(color='red', size=12, symbol='triangle-down'),
                                         textposition="bottom center"))

        fig.update_layout(title=f"{ticker} Candlestick Chart",
                          xaxis_rangeslider_visible=False, template='plotly_white',
                          yaxis=dict(title="Price"),
                          yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("MACD & Signal")
        st.line_chart(macd_line.rename("MACD").to_frame().join(signal_line.rename("Signal")))
        st.subheader("RSI")
        st.line_chart(hist["RSI"])
        st.subheader("ZigZag Pattern")
        st.line_chart(hist[["Close", "ZigZag"]])

    with tab3:
        st.markdown(f"**Recommendation:** {signal}")
        st.markdown(f"**Reason:** {reason}")
        st.markdown("**Indicator Knowledge Base:**")
        st.write("- **SMA20 & SMA50** show trend direction")
        st.write("- **MACD** indicates momentum shifts via moving average crossover")
        st.write("- **RSI** shows overbought/oversold levels")
        st.write("- **ZigZag** highlights price turning points")

    with tab4:
        if use_websearch:
            st.subheader("DuckDuckGo News")
            news_articles = fetch_duckduckgo_news(f"{ticker} CSE news")
            for n in news_articles:
                st.write(f"- [{n['title']}]({n['url']})")
        st.subheader("EconomyNext Headlines")
        econ_news = scrape_economynext()
        for hn in econ_news:
            st.write(f"- {hn}")

# ---------- FOOTER ----------
st.markdown("<div class='footer'>CSE Stock Dashboard – Powered by Streamlit & Plotly</div>", unsafe_allow_html=True)

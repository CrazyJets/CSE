# ------------------------------------------------------------------
# STEP 1: Install Libraries
# Run this command in your terminal or a Colab cell before running the script:
# pip install streamlit yfinance pandas numpy plotly scikit-learn wikipedia duckduckgo-search
# ------------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, UTC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import wikipedia
import requests
from bs4 import BeautifulSoup
import io
import PyPDF2
import json
import base64 # Import base64 for image embedding
warnings.filterwarnings('ignore')

# --- Helper function for image embedding (Streamlit specific) ---
def get_base64_image(image_path):
    """Encodes a local image file to a base64 string for Streamlit markdown."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        # Fallback to a placeholder base64 string or an empty string if image is not found
        # Using a small, light, transparent square image as a placeholder for robustness
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# --- Web Search Proxy Function ---
@st.cache_data(show_spinner="📡 Searching Wikipedia...")
def fetch_web_news(symbol):
    """Fetch general information using the Wikipedia library as a web search proxy."""
    try:
        search_term = f"{symbol} stock Sri Lanka"
        search_results = wikipedia.search(search_term, results=1)
        if not search_results:
            return {
                'search_information': {'total_results': 0},
                'results': []
            }

        page = wikipedia.page(search_results[0], auto_suggest=False)
        summary = page.summary.split('\n')[0][:500] + "..." if page.summary else "Summary not available."

        return {
            'search_information': {'total_results': 1},
            'results': [{
                'title': page.title,
                'url': page.url,
                'snippet': summary
            }]
        }
    except wikipedia.exceptions.PageError:
        return {
            'search_information': {'total_results': 0},
            'results': [],
            'error': "No Wikipedia article found for this symbol."
        }
    except Exception as e:
        return {
            'search_information': {'total_results': 0},
            'results': [],
            'error': f"Error fetching Wikipedia data: {str(e)}"
        }

# --- EconomyNext Market News Function ---
@st.cache_data(show_spinner="Fetching Market News...")
def fetch_economynext(max_results=10):
    """Fetch market news articles from EconomyNext's markets page."""
    try:
        results = []
        market_url = 'https://economynext.com/markets/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch market news page
        resp = requests.get(market_url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Get all article elements
        articles = soup.select('article.post')[:max_results]
        
        for article in articles:
            try:
                # Extract article details
                title_el = article.select_one('h2.entry-title, h3.entry-title')
                link_el = article.select_one('a')
                date_el = article.select_one('.entry-date')
                excerpt_el = article.select_one('.entry-content, .entry-summary') or article.select_one('p')
                
                # Get text content
                title = title_el.get_text(strip=True) if title_el else 'No title'
                link = link_el['href'] if link_el and link_el.has_attr('href') else market_url
                date = date_el.get_text(strip=True) if date_el else 'Date not available'
                snippet = excerpt_el.get_text(strip=True)[:300] if excerpt_el else ''
                
                # Add to results
                results.append({
                    'title': title,
                    'url': link,
                    'date': date,
                    'snippet': snippet,
                    'source': 'EconomyNext Markets'
                })
            except Exception as article_error:
                print(f"Error processing article: {article_error}")
                continue
                
        return results
    except Exception as e:
        print(f"Error fetching EconomyNext market news: {e}")
        return []


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file-like object"""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for p in reader.pages:
            page_text = p.extract_text() or ''
            text.append(page_text)
        return '\n'.join(text)
    except Exception as e:
        return f"[Error extracting PDF text: {e}]"


def process_uploaded_files(files):
    """Process uploaded xlsx/csv/pdf files and return a list of dicts with summaries"""
    processed = []
    for f in files or []:
        name = f.name
        ext = name.split('.')[-1].lower()
        if ext in ('xlsx', 'xls'):
            try:
                df = pd.read_excel(f)
                summary = df.head(5).to_csv(index=False)
                processed.append({'filename': name, 'type': 'spreadsheet', 'content': df, 'summary': summary})
            except Exception as e:
                processed.append({'filename': name, 'type': 'spreadsheet', 'content': None, 'summary': f"Error reading spreadsheet: {e}"})
        elif ext == 'csv':
            try:
                df = pd.read_csv(f)
                summary = df.head(5).to_csv(index=False)
                processed.append({'filename': name, 'type': 'spreadsheet', 'content': df, 'summary': summary})
            except Exception as e:
                processed.append({'filename': name, 'type': 'spreadsheet', 'content': None, 'summary': f"Error reading CSV: {e}"})
        elif ext == 'pdf':
            try:
                # Seek to start in case Streamlit has read the file
                f.seek(0)
                text = extract_text_from_pdf(f)
                # Use first 1000 chars as summary
                processed.append({'filename': name, 'type': 'pdf', 'content': text, 'summary': text[:1000]})
            except Exception as e:
                processed.append({'filename': name, 'type': 'pdf', 'content': None, 'summary': f"Error reading PDF: {e}"})
        else:
            processed.append({'filename': name, 'type': 'unknown', 'content': None, 'summary': 'Unsupported file type'})

    return processed


# --- Core StockAnalyzer Class (Modified for Streamlit Caching) ---
class StockAnalyzer:
    def __init__(self):
        # We'll re-initialize these within the function decorated with @st.cache_data
        # to ensure proper state management across Streamlit reruns.
        pass

    @st.cache_data(show_spinner="📡 Fetching stock data...")
    def fetch_stock_data(_self, symbol, period="1y"):
        """Fetch stock data with CSE-specific error handling and default suffix."""
        # Note: _self is used to indicate this method doesn't rely on StockAnalyzer state for this cached run.

        # Logic to append the correct CSE suffix for yfinance (e.g., 'ALUM' -> 'ALUM-N0000.CM')
        base_symbol = symbol.upper().split('-')[0].split('.')[0]
        symbol_with_suffix = f"{base_symbol}-N0000.CM"

        try:
            stock = yf.Ticker(symbol_with_suffix)
            data = stock.history(period=period)
            info = stock.info
            dividends = stock.dividends.reset_index() if stock.dividends is not None else pd.DataFrame(columns=['Date', 'Dividends'])
            dividends['Date'] = pd.to_datetime(dividends['Date']).dt.strftime('%Y-%m-%d')


            # Check for valid data
            if data.empty or info.get('regularMarketPrice') is None:
                # Try the symbol without the '-N0000' part if the full one fails (e.g., 'ALUM.CM')
                stock_alt = yf.Ticker(f"{base_symbol}.CM")
                data_alt = stock_alt.history(period=period)
                info_alt = stock_alt.info
                dividends_alt = stock_alt.dividends.reset_index() if stock_alt.dividends is not None else pd.DataFrame(columns=['Date', 'Dividends'])
                dividends_alt['Date'] = pd.to_datetime(dividends_alt['Date']).dt.strftime('%Y-%m-%d')

                if not data_alt.empty:
                    return data_alt, info_alt, dividends_alt, f"{base_symbol}.CM"

                raise ValueError(f"No market data or basic info found for {symbol_with_suffix} or {base_symbol}.CM")

            return data, info, dividends, symbol_with_suffix
        except Exception as e:
            # Raise Streamlit error instead of Gradio error
            raise Exception(f"❌ Error fetching data for {symbol}: {str(e)}. Try a valid CSE code like ALUM or HAYL.")

    # Apply caching to computationally intensive indicator calculation
    @st.cache_data(show_spinner="Calculating technical indicators...")
    def calculate_technical_indicators(_self, data):
        """Calculate comprehensive technical indicators using pure pandas/numpy and add basic signals"""
        df = data.copy()

        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Avoid division by zero, replace with a tiny number
        rs = gain / loss.replace(0, np.nan).fillna(1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # --- Basic Buying/Selling Signals ---
        # Initialize with 0 (Hold)
        df['Signal'] = 0

        # SMA Crossover Signal
        df['SMA_20_prev'] = df['SMA_20'].shift(1)
        df['SMA_50_prev'] = df['SMA_50'].shift(1)
        # Buy: 20-day crosses above 50-day
        df.loc[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20_prev'] <= df['SMA_50_prev']), 'Signal'] = 1
        # Sell: 20-day crosses below 50-day
        df.loc[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20_prev'] >= df['SMA_50_prev']), 'Signal'] = -1

        # RSI Signal
        # Buy (Oversold)
        df.loc[df['RSI'] < 30, 'Signal'] = 1
        # Sell (Overbought)
        df.loc[df['RSI'] > 70, 'Signal'] = -1

        # MACD Crossover Signal
        df['MACD_prev'] = df['MACD'].shift(1)
        df['MACD_signal_prev'] = df['MACD_signal'].shift(1)
        # Buy: MACD crosses above Signal
        df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD_prev'] <= df['MACD_signal_prev']), 'Signal'] = 1
        # Sell: MACD crosses below Signal
        df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD_prev'] >= df['MACD_signal_prev']), 'Signal'] = -1

        # Clean up temp columns
        df = df.drop(columns=['SMA_20_prev', 'SMA_50_prev', 'MACD_prev', 'MACD_signal_prev'], errors='ignore')

        return df

    def prepare_ml_features(_self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        df = df.dropna()

        # Returns and momentum
        df['Returns'] = df['Close'].pct_change()

        # Lag features
        for lag in [1, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Rolling statistics
        for window in [10, 20]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()

        # Add key indicators as features
        for col in ['RSI', 'MACD', 'Signal']:
            if col in df.columns:
                df[col] = df[col] # Copy to ensure feature is present

        return df.dropna()

    @st.cache_data(show_spinner="Training ML prediction model...")
    def train_prediction_model(_self, data):
        """Train ML model for price prediction"""
        # Re-initialize model/scaler within this cached function
        scaler = StandardScaler()
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Data preparation
        # Indicators must be re-calculated if not passed in as a cached result
        # Assuming `data` passed in here is already the output of calculate_technical_indicators
        df = _self.prepare_ml_features(data)

        if len(df) < 50:
            return None, None # Return None for both model info and scaler

        # Target: Predict next day's close
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        if len(df) < 50:
            return None, None

        # Features for prediction (excluding target-related and raw price columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                        'Returns', 'Target', 'BB_middle', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if not (col.startswith('SMA') and len(col) > 6)]

        X = df[feature_cols].select_dtypes(include=np.number)
        y = df['Target']

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 50 or X.empty:
            return None, None

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)

        test_score = model.score(X_test_scaled, y_test)

        model_info = {
            'test_score': test_score,
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            # Use only the last row of features for prediction
            'last_features': X.iloc[-1:].to_dict('records')[0],
            'feature_cols': X.columns.tolist(),
            'model': model
        }

        return model_info, scaler # Return both info and the fitted scaler


    def predict_next_price(_self, model_info, scaler):
        """Predict next trading day price"""
        if model_info is None or scaler is None:
            return None

        # Reconstruct the feature DataFrame for the last day
        last_features_df = pd.DataFrame([model_info['last_features']])

        # Ensure features are in the correct order and only include what was trained
        feature_cols = model_info['feature_cols']
        last_features = last_features_df[feature_cols]

        last_features_scaled = scaler.transform(last_features)
        prediction = model_info['model'].predict(last_features_scaled)[0]

        return prediction

    def generate_market_analysis(_self, data, info, symbol):
        """Generate AI-powered market analysis"""
        if len(data) < 2:
            return ["Insufficient data for analysis."]

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100

        # Technical analysis (handle missing columns if data is too short for indicators)
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', latest['Close'])
        signal = latest.get('Signal', 0)

        # Generate analysis
        analysis = []

        # Price trend
        if price_change_pct > 2:
            analysis.append(f"🚀 **Price Surge**: {symbol} shows strong bullish momentum with a **{price_change_pct:.2f}%** surge.")
        elif price_change_pct < -2:
            analysis.append(f"🔻 **Sharp Drop**: {symbol} faces significant selling pressure (**{price_change_pct:.2f}%**).")
        else:
            analysis.append(f"🟡 **Neutral Trend**: {symbol} saw modest movement (**{price_change_pct:+.2f}%**).")

        # Buying/Selling Signal
        if signal == 1:
            analysis.append("🟢 **Technical Signal**: Indicators suggest a **BUY** opportunity.")
        elif signal == -1:
            analysis.append("🔴 **Technical Signal**: Indicators suggest a **SELL** signal.")
        else:
            analysis.append("⚫ **Technical Signal**: Indicators are currently **NEUTRAL**.")

        # RSI analysis
        if 'RSI' in data.columns and not pd.isna(rsi):
            if rsi > 70:
                analysis.append(f"⚠️ **RSI Warning**: RSI at **{rsi:.1f}** suggests **overbought** conditions.")
            elif rsi < 30:
                analysis.append(f"🛒 **RSI Buy Signal**: RSI at **{rsi:.1f}** signals **oversold** conditions.")
            else:
                analysis.append(f"⚖️ **RSI Balanced**: RSI at **{rsi:.1f}** indicates **balanced** momentum.")

        # Moving average analysis
        if latest['Close'] > sma_20:
            analysis.append("📈 **Short-Term Trend**: Price is **above** the 20-day SMA, confirming a short-term bullish trend.")
        elif latest['Close'] < sma_20:
            analysis.append("📉 **Short-Term Trend**: Price is **below** the 20-day SMA, signaling bearish pressure.")

        # Market cap context
        market_cap = info.get('marketCap', 0)
        if market_cap:
            currency = info.get('financialCurrency', 'LKR')
            cap_type = "Large-cap" if market_cap > 50e9 else "Mid-cap" if market_cap > 5e9 else "Small-cap"
            analysis.append(f"🏢 **Context**: A **{cap_type}** stock in the {info.get('sector', 'N/A')} sector of the CSE with a market cap of approximately {market_cap/1e9:.1f}B {currency}.")


        return analysis

# --- Streamlit Plotly Chart Functions ---

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""

    if len(data) < 20:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f'{symbol} Price Chart (Insufficient Data for TIs)', template='plotly_dark')
        return fig

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # 1. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )

    # Moving averages (20)
    if 'SMA_20' in data.columns and not data['SMA_20'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'],
                      line=dict(color='#ff9500', width=1.5), name='SMA 20'),
            row=1, col=1
        )

    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)

    # 2. MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='#007aff', width=2), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], line=dict(color='#ff9500', width=2), name='Signal'), row=2, col=1)
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_histogram'], marker_color=histogram_colors, name='Histogram', opacity=0.6), row=2, col=1)

    # 3. RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='#af52de', width=2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,
        template='plotly_dark',
        font=dict(size=10),
        height=700 # Explicitly set height for better Streamlit display
    )

    for i in range(1, 3):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    return fig

# --- Main Streamlit App Logic ---

def main():
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="🇱🇰 AI Stock Dashboard (CSE)",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # --- Title and Description ---
    st.title("🇱🇰 AI Stock Market Dashboard (CSE Focused)")
    st.markdown("Analyze Colombo Stock Exchange (CSE) data, generate technical charts, and predict prices using an AI model. **Try stock codes like ALUM, HAYL, or DIST.**")
    st.divider()

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("Stock Selection")
        stock_symbol_input = st.text_input(
            "CSE Symbol",
            value="ALUM",
            placeholder="Enter stock code like ALUM or HAYL"
        ).upper()

        period_input = st.selectbox(
            "Analysis Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3 # Default to 1y
        )

        st.divider()
        st.caption("Data is fetched from Yahoo Finance. CSE symbols often require a suffix.")
        st.divider()
        st.header("Upload Financial Files")
        uploaded_files = st.file_uploader(
            "Upload XLSX, CSV or PDF files related to the ticker (optional)",
            type=['xlsx', 'xls', 'csv', 'pdf'],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")


    if not stock_symbol_input:
        st.warning("Please enter a stock symbol to begin analysis.")
        return

    # Initialize analyzer
    analyzer = StockAnalyzer()

    try:
        # 1. Fetch Data
        data, info, dividends, actual_symbol = analyzer.fetch_stock_data(stock_symbol_input, period_input)

        # 2. Calculate Indicators
        data = analyzer.calculate_technical_indicators(data)

        # 3. Get Latest Metrics
        latest_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        currency = info.get('financialCurrency', 'LKR')

        # --- Display Key Metrics as Columns ---
        st.header(f"Key Metrics for {info.get('longName', actual_symbol)} ({actual_symbol})")
        col_price, col_cap, col_rsi, col_sma = st.columns(4)

        with col_price:
            st.metric(
                label="Current Price",
                value=f"{currency} {latest_price:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )

        with col_cap:
            market_cap = info.get('marketCap', 0)
            formatted_cap = f"{market_cap/1e9:.1f}B {currency}" if market_cap else 'N/A'
            st.metric(
                label="Market Cap",
                value=formatted_cap,
                delta=info.get('sector', 'N/A'),
                delta_color="off"
            )

        with col_rsi:
            rsi_val = data['RSI'].iloc[-1]
            rsi_status = 'Overbought (SELL)' if rsi_val > 70 else 'Oversold (BUY)' if rsi_val < 30 else 'Neutral'
            st.metric(
                label="RSI (14)",
                value=f"{rsi_val:.1f}",
                delta=rsi_status,
                delta_color="inverse" if rsi_val > 70 or rsi_val < 30 else "off"
            )

        with col_sma:
            sma_20_val = data['SMA_20'].iloc[-1]
            sma_status = 'Above SMA' if latest_price > sma_20_val else 'Below SMA'
            st.metric(
                label="20-day SMA",
                value=f"{sma_20_val:.2f}",
                delta=sma_status,
                delta_color="normal" if latest_price > sma_20_val else "inverse"
            )

        st.divider()

        # --- Main Tabs Layout ---
        tab_chart, tab_analysis, tab_financials, tab_news = st.tabs(
            ["📈 Technical Chart & Prediction", "🧠 AI Analysis", "📊 Financials & Dividends", "📰 Recent News"]
        )

        with tab_chart:
            # Chart and ML Prediction in the Chart Tab

            col_chart, col_prediction = st.columns([3, 1])

            with col_chart:
                st.subheader("Advanced Technical Analysis Chart")
                chart = create_advanced_chart(data, actual_symbol)
                st.plotly_chart(chart, use_container_width=True)

            with col_prediction:
                st.subheader("ML Prediction")
                # 4. ML Prediction and Feature Importance
                model_info, scaler = analyzer.train_prediction_model(data)

                if model_info:
                    prediction = analyzer.predict_next_price(model_info, scaler)
                    confidence = model_info['test_score']
                    predicted_change = ((prediction - latest_price) / latest_price) * 100

                    st.markdown("#### Next Day Price Forecast")
                    st.metric(
                        label="Predicted Close Price",
                        value=f"{currency} {prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )

                    st.markdown(f"**Model Accuracy (R-sq):** `{confidence:.1%}`")
                    st.caption("Random Forest Regressor attempts to predict the next day's closing price based on calculated indicators.")

                    st.markdown("---")
                    st.markdown("#### Feature Importance")
                    feature_importance_df = pd.DataFrame(
                        list(model_info['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False).head(10)

                    fig_importance = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 ML Features",
                        template='plotly_dark'
                    )
                    fig_importance.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_importance, use_container_width=True)

                else:
                    st.warning("Insufficient data to train the Machine Learning model.")


        with tab_analysis:
            st.subheader("🧠 AI-Powered Market Analysis")
            # 5. AI Market Analysis
            analysis_insights = analyzer.generate_market_analysis(data, info, actual_symbol)
            analysis_markdown = ""
            for insight in analysis_insights:
                analysis_markdown += f"* {insight}\n"
            st.markdown(analysis_markdown)


        with tab_financials:
            col_financials, col_dividends = st.columns(2)

            with col_financials:
                st.subheader("📊 Financial Highlights")
                # 8. Financial Highlights
                financial_metrics = {
                    "P/E Ratio": info.get('trailingPE'),
                    "Forward P/E": info.get('forwardPE'),
                    "PEG Ratio": info.get('pegRatio'),
                    "Dividend Yield": info.get('dividendYield'),
                    "EPS (Trailing)": info.get('trailingEPS'),
                    "EPS (Forward)": info.get('forwardEPS'),
                    "Book Value": info.get('bookValue'),
                    "Price to Book": info.get('priceToBook'),
                    "Market Cap": info.get('marketCap'),
                    "Volume (Avg 10d)": info.get('averageVolume10days'),
                    "52 Week High": info.get('fiftyTwoWeekHigh'),
                    "52 Week Low": info.get('fiftyTwoWeekLow')
                }

                financial_data = []
                for metric, value in financial_metrics.items():
                    if value is not None:
                        if 'Yield' in metric:
                             formatted_value = f"{value:.2%}" if isinstance(value, (int, float)) else value
                        elif 'Volume' in metric or 'Cap' in metric:
                             formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else value
                        elif isinstance(value, (int, float)):
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = value
                        financial_data.append({'Metric': metric, 'Value': formatted_value})

                if financial_data:
                    st.dataframe(pd.DataFrame(financial_data).set_index('Metric'), use_container_width=True)
                else:
                    st.info("* No detailed financial highlights available.")


            with col_dividends:
                st.subheader("💰 Dividend History")
                # 7. Dividend Information
                if not dividends.empty:
                    # Rename columns for clarity in Streamlit
                    dividends.columns = ['Ex-Date', 'Dividends']
                    st.dataframe(dividends, use_container_width=True)
                else:
                    st.info("* No dividend history found for this period.")


        with tab_news:
            st.subheader("📰 Latest Market News")
            
            # Process uploaded files
            processed = process_uploaded_files(uploaded_files) if 'uploaded_files' in locals() or uploaded_files else []
            if processed:
                st.markdown("### 📁 Uploaded Files")
                for p in processed:
                    st.markdown(f"**{p['filename']}** — {p['type']}")
                    if p['type'] == 'spreadsheet' and isinstance(p['content'], pd.DataFrame):
                        st.dataframe(p['content'].head(10), use_container_width=True)
                    else:
                        st.text(p['summary'])
                    st.markdown('---')

            # Fetch EconomyNext market news
            st.markdown("### 📈 Market News")
            market_news = fetch_economynext()
            if market_news:
                for article in market_news:
                    with st.expander(article['title'], expanded=False):
                        st.markdown(f"**Date**: {article['date']}")
                        st.markdown(f"**Summary**: {article['snippet']}")
                        st.markdown(f"**Read More**: [EconomyNext]({article['url']})")
            else:
                st.info("Unable to fetch market news at this time. Please try again later.")
                # Add Wikipedia information as additional context
                st.markdown("### 📚 Additional Information")
                wiki_results = fetch_web_news(stock_symbol_input)
                if wiki_results.get('results'):
                    for result in wiki_results['results']:
                        with st.expander(result['title'], expanded=True):
                            st.markdown(f"**Summary**: {result['snippet']}")
                            st.markdown(f"**Source**: [Wikipedia]({result['url']})")
                else:
                    st.info("No additional information found for this symbol.")

    except Exception as e:
        # Catch and display the error from fetch_stock_data
        st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()

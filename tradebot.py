import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from dash.dependencies import Input, Output, State
import talib as talib
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from flask import Flask, render_template
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from datetime import datetime, timedelta
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import warnings
import statsmodels.api as sm
import dash

app = dash.Dash(__name__)
server = app.server 
PORT = int(os.environ.get('PORT', 8050))
# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

class Attention(Layer):
    """
    Attention layer for LSTM
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                               shape=(input_shape[-1], 1),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='attention_bias',
                               shape=(input_shape[1], 1),
                               initializer='zeros',
                               trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
# Fetch stock data
def fetch_stock_data(ticker, period="3mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# News Sentiment Analysis Functions
def fetch_stock_news(ticker, max_articles=10):
    """Fetch news articles about a stock from multiple sources"""
    try:
        all_news = []
        
        # Get news from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        yahoo_news = ticker_obj.news
        if yahoo_news:
            for article in yahoo_news:
                all_news.append({
                    'title': article.get('title', ''),
                    'link': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M'),
                    'summary': article.get('summary', '')
                })

        # Get news from Finviz
        try:
            base_url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(base_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find(id='news-table')
            if news_table:
                rows = news_table.findAll('tr')
                for row in rows:
                    title_link = row.a
                    if title_link:
                        date_td = row.td
                        if date_td:
                            date_str = date_td.text.strip()
                            all_news.append({
                                'title': title_link.text.strip(),
                                'link': title_link['href'],
                                'published': date_str,
                                'summary': ''
                            })
        except Exception as e:
            print(f"Error fetching Finviz news: {e}")

        # Remove duplicates based on title
        seen_titles = set()
        unique_news = []
        for article in all_news:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_news.append(article)

        # Sort by publication date (if available)
        try:
            unique_news.sort(key=lambda x: datetime.strptime(x['published'].split(' ')[0], '%Y-%m-%d'), reverse=True)
        except:
            pass  # If date parsing fails, keep original order

        return unique_news[:max_articles]
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_news_sentiment(news_articles):
    """Analyze sentiment from news articles with enhanced analysis"""
    if not news_articles:
        return {
            'score': 50,  # Neutral score when no news available
            'pros': ["No recent news analysis available"],
            'cons': ["No recent news analysis available"],
            'risks': ["Unable to determine risks without news data"]
        }
    
    # Enhanced keyword lists
    risk_keywords = {
        'high': ['lawsuit', 'investigation', 'fraud', 'recall', 'breach', 'violation', 'crash'],
        'medium': ['risk', 'warning', 'concern', 'volatility', 'uncertainty'],
        'low': ['drop', 'fall', 'decline', 'decrease', 'lower']
    }
    
    pos_keywords = {
        'high': ['breakthrough', 'exceptional', 'surpass', 'strongest', 'record high', 'patent'],
        'medium': ['growth', 'profit', 'beat', 'upgrade', 'innovation', 'partnership'],
        'low': ['rise', 'gain', 'positive', 'opportunity', 'improve']
    }
    
    neg_keywords = {
        'high': ['bankruptcy', 'layoff', 'default', 'scandal', 'disaster'],
        'medium': ['loss', 'downgrade', 'fail', 'debt', 'missed', 'below'],
        'low': ['decline', 'decrease', 'slow', 'challenge', 'concern']
    }
    
    # Initialize sentiment tracking
    sentiment_scores = []
    pros = []
    cons = []
    risks = []
    
    # Analyze each article
    for article in news_articles:
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        full_text = f"{title} {summary}"
        
        # Perform sentiment analysis on full text
        analysis = TextBlob(full_text)
        
        # Weight the sentiment score
        sentiment_score = analysis.sentiment.polarity
        sentiment_scores.append(sentiment_score)
        
        # Check for keywords with different weights
        risk_score = 0
        pos_score = 0
        neg_score = 0
        
        # Calculate keyword scores
        for level, keywords in risk_keywords.items():
            weight = 3 if level == 'high' else 2 if level == 'medium' else 1
            if any(keyword in full_text for keyword in keywords):
                risk_score += weight
        
        for level, keywords in pos_keywords.items():
            weight = 3 if level == 'high' else 2 if level == 'medium' else 1
            if any(keyword in full_text for keyword in keywords):
                pos_score += weight
        
        for level, keywords in neg_keywords.items():
            weight = 3 if level == 'high' else 2 if level == 'medium' else 1
            if any(keyword in full_text for keyword in keywords):
                neg_score += weight
        
        # Categorize article based on scores and sentiment
        if risk_score > 0:
            risks.append(article.get('title'))
        
        if pos_score > neg_score or sentiment_score > 0.2:
            pros.append(article.get('title'))
        elif neg_score > pos_score or sentiment_score < -0.2:
            cons.append(article.get('title'))
    
    # Calculate final sentiment score
    if sentiment_scores:
        avg_sentiment = np.mean(sentiment_scores)
        base_score = int((avg_sentiment + 1) * 50)  # Convert from [-1,1] to [0,100]
        
        # Adjust score based on pros/cons ratio
        pros_weight = len(pros) / (len(pros) + len(cons)) if pros or cons else 0.5
        final_score = int(base_score * 0.7 + (pros_weight * 100) * 0.3)
        final_score = max(0, min(100, final_score))  # Ensure score is between 0 and 100
    else:
        final_score = 50
    
    # Ensure we have unique items and sort by relevance
    pros = list(dict.fromkeys(pros))  # Remove duplicates
    cons = list(dict.fromkeys(cons))
    risks = list(dict.fromkeys(risks))
    
    # Provide default messages if no items found
    pros = pros[:3] if pros else ["No significant positive factors identified in recent news"]
    cons = cons[:3] if cons else ["No significant negative factors identified in recent news"]
    risks = risks[:3] if risks else ["No clear risks identified in recent news"]
    
    return {
        'score': final_score,
        'pros': pros,
        'cons': cons,
        'risks': risks
    }

# Fix: Improved LSTM prediction function with error handling
def predict_with_lstm(data, future_days=10):
    if data.empty or len(data) < 40:
        print(f"Not enough data for LSTM prediction. Have {len(data)} points, need at least 40.")
        return None, None
    
    try:
        # Prepare data
        close_data = data['Close'].values.reshape(-1, 1)
        look_back = min(30, len(data) - 10)  # Adaptive look-back period
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)
        
        # Prepare training data
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        # Convert to numpy arrays and reshape
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to 3D for LSTM input
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Create and train the model with early stopping
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model with reduced epochs and early stopping
        model.fit(X, y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        # Prepare input for future prediction
        last_sequence = scaled_data[-look_back:]
        last_sequence = np.array(last_sequence).reshape(1, look_back, 1)
        future_predictions = []
        
        # Make sequential predictions
        curr_sequence = last_sequence.copy()
        for i in range(future_days):
            next_pred = model.predict(curr_sequence, verbose=0)[0][0]
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            curr_sequence = np.append(curr_sequence[:, 1:, :], 
                                     np.array([[next_pred]]).reshape(1, 1, 1),
                                     axis=1)
        
        # Inverse transform to get actual prices
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(future_predictions)
        
        # Adjust predictions to start from the last real price
        last_real_price = close_data[-1, 0]
        predicted_prices = predicted_prices.flatten()
        adjustment = last_real_price - predicted_prices[0]
        predicted_prices = predicted_prices + adjustment
        
        # Create dates for prediction
        last_date = data.index[-1]
        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        return prediction_dates, predicted_prices
    
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        return None, None

# Fix: Improved ARIMA simulation with better error handling
def simulate_stock(data):
    if data.empty or len(data) < 10:
        print("Not enough data to simulate with ARIMA.")
        return pd.Series([], dtype=float)

    try:
        data = data.copy()
        # Fix for non-business days in index
        data.index = pd.to_datetime(data.index)
        
        # Use log returns for better stationarity
        close_data = data['Close']
        log_returns = np.log(close_data).diff().dropna()
        
        # Fit ARIMA model with lower max order to avoid overfitting
        model = auto_arima(
            log_returns, 
            seasonal=False, 
            stepwise=True,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            max_d=1,
            trace=False,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Forecast returns
        forecast_returns = model.predict(n_periods=10)
        
        # Convert returns to prices
        last_price = close_data.iloc[-1]
        forecast_prices = [last_price]
        for ret in forecast_returns:
            next_price = forecast_prices[-1] * np.exp(ret)
            forecast_prices.append(next_price)
        
        forecast_dates = pd.date_range(data.index[-1], periods=11, freq='D')[1:]
        return pd.Series(forecast_prices[1:], index=forecast_dates)
    
    except Exception as e:
        print(f"Error in ARIMA simulation: {e}")
        return pd.Series([], dtype=float)

# Fix: Improved ETS simulation
def simulate_stock_ets(data):
    if data.empty or len(data) < 10:
        print("Not enough data to simulate with ETS.")
        return pd.Series([], dtype=float)

    try:
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        close_data = data['Close']
        
        # Determine if seasonality should be used based on data length
        seasonal = None
        seasonal_periods = None
        if len(close_data) >= 60:  # Enough data for weekly seasonality
            seasonal = "add"
            seasonal_periods = 5  # Business week
        
        model = ExponentialSmoothing(
            close_data, 
            trend="add", 
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped=True
        )
        model_fit = model.fit(optimized=True)
        forecast = model_fit.forecast(steps=10)
        
        forecast_dates = pd.date_range(data.index[-1], periods=11, freq='D')[1:]
        return pd.Series(forecast.values, index=forecast_dates)
    
    except Exception as e:
        print(f"Error in ETS simulation: {e}")
        return pd.Series([], dtype=float)

# Improved technical indicators with error handling
def add_indicators(data):
    if data.empty:
        return data
    
    try:
        result = data.copy()
        # Calculate the RSI (Relative Strength Index)
        result['RSI'] = talib.RSI(result['Close'], timeperiod=14)

        # Calculate the MACD (Moving Average Convergence Divergence)
        result['MACD'], result['MACD_signal'], _ = talib.MACD(
            result['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Calculate moving averages (Simple and Exponential)
        result['SMA_10'] = talib.SMA(result['Close'], timeperiod=10)
        result['SMA_50'] = talib.SMA(result['Close'], timeperiod=50)
        result['EMA_10'] = talib.EMA(result['Close'], timeperiod=10)
        result['EMA_50'] = talib.EMA(result['Close'], timeperiod=50)

        # Calculate Bollinger Bands
        result['upper_band'], result['middle_band'], result['lower_band'] = talib.BBANDS(
            result['Close'], timeperiod=20
        )
        
        return result
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return data

def get_buy_sell_signals(data):
    if data.empty:
        return []
        
    signals = []
    try:
        # Ensure we have all required indicators
        required_cols = ['RSI', 'MACD', 'MACD_signal', 'Close', 'SMA_10', 'SMA_50', 'upper_band', 'lower_band']
        if not all(col in data.columns for col in required_cols):
            return ["Insufficient indicator data for signals"]
        
        # Get last values, handling NaN values
        last_rsi = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50
        last_macd = data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0
        last_macd_signal = data['MACD_signal'].iloc[-1] if not pd.isna(data['MACD_signal'].iloc[-1]) else 0
        
        # Buy signals based on indicators
        if last_rsi < 30:
            signals.append('Buy - RSI (Oversold)')
        if last_macd > last_macd_signal:
            signals.append('Buy - MACD (Bullish Crossover)')
        if data['Close'].iloc[-1] > data['SMA_10'].iloc[-1] > data['SMA_50'].iloc[-1]:
            signals.append('Buy - Moving Averages (Uptrend)')
        if data['Close'].iloc[-1] < data['lower_band'].iloc[-1]:
            signals.append('Buy - Bollinger Bands (Below Lower Band)')

        # Sell signals based on indicators
        if last_rsi > 70:
            signals.append('Sell - RSI (Overbought)')
        if last_macd < last_macd_signal:
            signals.append('Sell - MACD (Bearish Crossover)')
        if data['Close'].iloc[-1] < data['SMA_10'].iloc[-1] < data['SMA_50'].iloc[-1]:
            signals.append('Sell - Moving Averages (Downtrend)')
        if data['Close'].iloc[-1] > data['upper_band'].iloc[-1]:
            signals.append('Sell - Bollinger Bands (Above Upper Band)')
    
    except Exception as e:
        print(f"Error generating signals: {e}")
        signals.append(f"Error generating signals: {str(e)}")
    
    return signals

# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# App Layout with News Sentiment Section
dashboard_layout = html.Div(
    style={'backgroundColor': '#f4f6f9', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.H1("Advanced Stock Analysis Dashboard", style={'textAlign': 'center', 'fontFamily': 'Helvetica, Arial', 'color': '#333'}),
        
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'padding': '10px 0'},
            children=[
                dcc.Input(id='ticker-input', type='text', placeholder='Enter stock ticker (e.g., AAPL)', 
                         style={'padding': '10px', 'width': '200px', 'fontSize': '16px', 'borderRadius': '5px'}),
                
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': '1 day', 'value': '1d'},
                        {'label': '5 days', 'value': '5d'},
                        {'label': '1 month', 'value': '1mo'},
                        {'label': '2 months', 'value': '2mo'},
                        {'label': '3 months', 'value': '3mo'},
                        {'label': '6 months', 'value': '6mo'},
                        {'label': '1 year', 'value': '1y'},
                        {'label': '5 years', 'value': '5y'}
                    ],
                    value='3mo',
                    style={'width': '200px', 'padding': '5px', 'fontSize': '16px', 'borderRadius': '5px'}
                )
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px', 'gap': '10px'},
            children=[
                html.Button('Fetch Data', id='fetch-button', n_clicks=0, 
                          style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                                'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('LSTM Prediction', id='lstm-button', n_clicks=0, 
                          style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                                'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('LSTM Advanced', id='lstm-advanced-button', n_clicks=0, 
                        style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                               'backgroundColor': '#9C27B0', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('ARIMA Simulation', id='arima-button', n_clicks=0, 
                          style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                                'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('ETS Simulation', id='ets-button', n_clicks=0, 
                          style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                                'backgroundColor': '#9C27B0', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('News Analysis', id='news-button', n_clicks=0, 
                          style={'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px', 
                                'backgroundColor': '#E91E63', 'color': 'white', 'border': 'none', 'cursor': 'pointer'})
            ]
        ),
        
        dcc.Graph(id='stock-graph', style={'marginTop': '30px'}),
        
        # Technical Analysis Output
        html.Div(id='prediction-output', 
                style={'textAlign': 'left', 'marginTop': '20px', 'fontSize': '16px', 'color': '#555', 
                      'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '5px', 
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # News Sentiment Analysis Section
        html.Div(
            id='news-sentiment-container',
            style={'display': 'none', 'marginTop': '30px'},
            children=[
                html.H2("News Sentiment Analysis", style={'fontSize': '20px', 'marginBottom': '15px'}),
                
                # Sentiment Score Display
                html.Div(
                    id='sentiment-score-container',
                    style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'},
                    children=[
                        html.H3("Market Sentiment Score:", style={'marginRight': '15px'}),
                        html.Div(
                            id='sentiment-score',
                            style={
                                'width': '60px', 
                                'height': '60px', 
                                'borderRadius': '50%', 
                                'display': 'flex', 
                                'justifyContent': 'center', 
                                'alignItems': 'center',
                                'fontWeight': 'bold',
                                'fontSize': '20px',
                                'color': 'white',
                                'backgroundColor': '#888'
                            },
                            children="--"
                        ),
                        html.Div(
                            style={'marginLeft': '15px', 'fontSize': '14px', 'color': '#666'},
                            children="(0-100 scale: higher is more positive)"
                        )
                    ]
                ),
                
                # News Analysis Grid
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'},
                    children=[
                        # Pros and Cons Section
                        html.Div(
                            children=[
                                html.Div(
                                    style={'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'},
                                    children=[
                                        html.H3("Positive Factors", style={'color': '#2e7d32', 'marginBottom': '10px'}),
                                        html.Div(id='pros-list', style={'color': '#1b5e20'})
                                    ]
                                ),
                                html.Div(
                                    style={'backgroundColor': '#ffebee', 'padding': '15px', 'borderRadius': '5px'},
                                    children=[
                                        html.H3("Negative Factors", style={'color': '#c62828', 'marginBottom': '10px'}),
                                        html.Div(id='cons-list', style={'color': '#b71c1c'})
                                    ]
                                )
                            ]
                        ),
                        
                        # Risks Section
                        html.Div(
                            style={'backgroundColor': '#fff8e1', 'padding': '15px', 'borderRadius': '5px'},
                            children=[
                                html.H3("Potential Risks", style={'color': '#ff6f00', 'marginBottom': '10px'}),
                                html.Div(id='risks-list', style={'color': '#e65100'})
                            ]
                        )
                    ]
                ),
                
                # News Sources
                html.Div(
                    id='news-sources-container',
                    style={'marginTop': '30px'},
                    children=[
                        html.H3("Recent News", style={'marginBottom': '10px'}),
                        html.Div(id='news-sources-list', style={'fontSize': '14px'})
                    ]
                )
            ]
        )
    ]
)
# First, create the landing page layout
landing_page_layout = html.Div(
    className='aurora-background',
    style={
        'height': '100vh',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        # Aurora effect container
        html.Div(
            className='aurora-container',
            children=[
                html.Div(className='aurora')
            ]
        ),
        # Your existing content
        html.Div(
            style={
                'textAlign': 'center',
                'padding': '50px 20px',
                'position': 'relative',
                'zIndex': '1',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'center',
                'height': '100vh'
            },
            children=[
                html.Div(
                    style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'maxWidth': 'fit-content',
                        'marginTop': '20px'  # Reduced margin to move content up
                    },
                    children=[
                        html.H1(
                            "AI Stock Prediction Platform.",
                            style={
                                'fontSize': '3.5rem',
                                'fontWeight': '800',
                                'color': '#1e3a8a',
                                'marginBottom': '20px',
                                'background': 'linear-gradient(to right, #1e3a8a, #6366f1)',
                                '-webkit-background-clip': 'text',
                                '-webkit-text-fill-color': 'transparent'
                            },
                            className='typing-animation'
                        ),
                        html.H3(
                            "Advanced market analysis using LSTM and Neural Networks",
                            style={
                                'fontSize': '1.5rem',
                                'color': '#1f2937',
                                'marginBottom': '40px',
                                'maxWidth': '700px',
                                'textAlign': 'center'
                            }
                        ),
                        dcc.Link(
                            html.Button(
                                [
                                    html.Div(className='animated-button-glow'),
                                    html.Span("📈", className="button-emoji"),
                                    html.Span("Enter Dashboard", className="button-text"),
                                ],
                                className="animated-button",
                                id='enter-dashboard',
                                style={
                                    'fontSize': '16px',  # Adjust font size
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center'
                                }
                            ),
                            href='/dashboard',
                            style={'textDecoration': 'none', 'marginBottom': '50px'}
                        ),
                        # Feature highlights with improved font
                        html.Div(
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'gap': '30px',
                                'marginTop': '50px',
                                'flexWrap': 'wrap'
                            },
                            className='feature-highlights',
                            children=[
                                html.Div(
                                    className='feature-card',
                                    style={
                                        'fontFamily': 'Inter, sans-serif',  # More modern, clean font
                                        'fontWeight': '700'
                                    },
                                    children=[
                                        html.H4("LSTM Predictions", style={'color': '#1e3a8a', 'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}),
                                        html.P("Advanced neural network predictions with sophisticated confidence intervals")
                                    ]
                                ),
                                html.Div(
                                    className='feature-card',
                                    style={
                                        'fontFamily': 'Inter, sans-serif',
                                        'fontWeight': '700'
                                    },
                                    children=[
                                        html.H4("Technical Analysis", style={'color': '#1e3a8a', 'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}),
                                        html.P("Comprehensive technical indicators and market analysis")
                                    ]
                                ),
                                html.Div(
                                    className='feature-card',
                                    style={
                                        'fontFamily': 'Inter, sans-serif',
                                        'fontWeight': '700'
                                    },
                                    children=[
                                        html.H4("Market Sentiment", style={'color': '#1e3a8a', 'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}),
                                        html.P("Real-time sentiment analysis and news impact")
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
       
        /* Existing Aurora Styles */
        .aurora-background {
            position: relative;
            min-height: 100vh;
            background-color: #f4f6f9;
            overflow: hidden;
        }

        .aurora-container {
            position: absolute;
            inset: -10px;
            overflow: hidden;
            opacity: 0.5;
            pointer-events: none;
        }

        .aurora {
            position: absolute;
            inset: 0;
            background-image: 
                repeating-linear-gradient(100deg, #fff 0%, #fff 7%, transparent 10%, transparent 12%, #fff 16%),
                repeating-linear-gradient(100deg, #3b82f6 10%, #a5b4fc 15%, #93c5fd 20%, #ddd6fe 25%, #60a5fa 30%);
            background-size: 300% 200%;
            background-position: 50% 50%, 50% 50%;
            filter: blur(10px);
            animation: aurora 60s linear infinite;
            mask-image: radial-gradient(ellipse at 100% 0%, black 10%, transparent 70%);
        }

        @keyframes aurora {
            from {
                background-position: 50% 50%, 50% 50%;
            }
            to {
                background-position: 350% 50%, 350% 50%;
            }
        }

        /* Typing Animation */
        .typing-animation {
    overflow: hidden;
    white-space: nowrap;
    margin: 0 auto;
    animation: typing 3.5s steps(40, end);
    border-right: .15em solid #2c3e50;
    animation-iteration-count: 1;
    animation-fill-mode: forwards;
    
    /* New blinking cursor effect */
    animation-delay: 0s, 3.5s;
    animation-name: typing, blink-caret;
    animation-duration: 3.5s, 1.0s;
    animation-timing-function: steps(40, end), step-end;
    animation-iteration-count: 1, infinite;
}

@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

@keyframes blink-caret {
    0%, 100% { border-color: transparent }
    50% { border-color: #2c3e50 }
}

        /* Enhanced Design Additions */
        :root {
            --primary-color: #1e3a8a;
            --secondary-color: #4338ca;
            --accent-color: #6366f1;
        }

        /* Glassmorphic Feature Cards */
        .feature-highlights {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 50px;
            flex-wrap: wrap;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(15px);
            border-radius: 1rem;
            padding: 20px;
            width: 250px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(30, 58, 138, 0.05);
        }

        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 35px rgba(30, 58, 138, 0.1);
        }

        .feature-card h4 {
            color: var(--primary-color);
            margin-bottom: 10px;
        }

        /* Animated Button Enhancements */
       /* Animated button styles */
.animated-button {
    position: relative;
    padding: 15px 30px;
    font-size: 16px;
    background: linear-gradient(to right, #6366f1, #4338ca);
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    overflow: hidden;  /* Keep overflow hidden */
    transition: all 0.3s ease;
    font-family: 'Inter', sans-serif;
    'fontWeight': '700'
    width: 200px;
    text-align: center;
    display: flex;
    align-items: center;
    z-index: 1;
}

.animated-button-glow {
    position: absolute;
    top: -15px;
    left: -15px;
    right: -15px;
    bottom: -15px;
    background: linear-gradient(
        45deg, 
        rgba(99, 102, 241, 0.5), 
        rgba(67, 56, 202, 0.5), 
        rgba(99, 102, 241, 0.5)
    );
    border-radius: 15px;
    opacity: 0;
    transition: opacity 0.3s ease;
    filter: blur(25px);
    z-index: -2;
    background-size: 400% 400%;
    animation: glowAnimation 3s ease infinite paused;
    pointer-events: none;
}

.button-text {
    display: inline-block;
    transition: transform 0.5s ease;
    white-space: nowrap;
}

.animated-button:hover .button-text {
    transform: translateX(200px);  /* Move text off the button */
}

.button-emoji {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%) translateX(-100px);
    transition: all 0.5s ease;
    font-size: 24px;
    opacity: 0;
}

.animated-button:hover .button-emoji {
    transform: translate(-50%, -50%) translateX(0);
    opacity: 1;
}

.animated-button:hover .animated-button-glow {
    opacity: 1;
    animation-play-state: running;
}

@keyframes glowAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .feature-highlights {
                flex-direction: column;
                align-items: center;
            }
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create a container for both layouts
app.layout = html.Div([
    html.H1("My Dash App")
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Simplified routing callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return landing_page_layout
    elif pathname == '/dashboard':
        return dashboard_layout
    else:
        return landing_page_layout
def generate_advanced_prediction(data, ticker):
    """
    Generate comprehensive trading recommendations with specific price targets and dates
    """
    try:
        # Initialize prediction components
        predictions = {
            'lstm': None,
            'arima': None,
            'ets': None,
            'support_levels': [],
            'resistance_levels': [],
            'price_targets': [],
            'confidence': 0,
            'recommendation': '',
            'entry_points': [],
            'exit_points': [],
            'timeframes': []
        }
        
        # Calculate Support and Resistance Levels
        def calculate_support_resistance(prices, window=20):
            highs = []
            lows = []
            for i in range(window, len(prices) - window):
                if all(prices[i] > prices[i-j] for j in range(1, window)) and \
                   all(prices[i] > prices[i+j] for j in range(1, window)):
                    highs.append(prices[i])
                if all(prices[i] < prices[i-j] for j in range(1, window)) and \
                   all(prices[i] < prices[i+j] for j in range(1, window)):
                    lows.append(prices[i])
            return np.array(lows), np.array(highs)

        # Get support and resistance levels
        lows, highs = calculate_support_resistance(data['Close'].values)
        if len(lows) > 0:
            predictions['support_levels'] = sorted(lows)[-3:]  # Top 3 support levels
        if len(highs) > 0:
            predictions['resistance_levels'] = sorted(highs)[-3:]  # Top 3 resistance levels

        # Get predictions from different models
        lstm_dates, lstm_prices = predict_with_lstm(data)
        if lstm_dates is not None:
            predictions['lstm'] = {
                'dates': lstm_dates,
                'prices': lstm_prices
            }

        arima_forecast = simulate_stock(data)
        if not arima_forecast.empty:
            predictions['arima'] = {
                'dates': arima_forecast.index,
                'prices': arima_forecast.values
            }

        ets_forecast = simulate_stock_ets(data)
        if not ets_forecast.empty:
            predictions['ets'] = {
                'dates': ets_forecast.index,
                'prices': ets_forecast.values
            }

        # Calculate Technical Indicators
        current_price = data['Close'].iloc[-1]
        rsi = talib.RSI(data['Close'].values)[-1]
        macd, signal, _ = talib.MACD(data['Close'].values)
        current_macd = macd[-1]
        current_signal = signal[-1]
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'].values)
        bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])

        # Generate Trading Signals
        signals = []
        confidence_factors = []

        # RSI Signals
        if rsi < 30:
            signals.append('buy')
            confidence_factors.append(0.8)
        elif rsi > 70:
            signals.append('sell')
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)

        # MACD Signals
        if current_macd > current_signal:
            signals.append('buy')
            confidence_factors.append(0.7)
        else:
            signals.append('sell')
            confidence_factors.append(0.7)

        # Bollinger Bands Signals
        if bb_position < 0.2:
            signals.append('buy')
            confidence_factors.append(0.6)
        elif bb_position > 0.8:
            signals.append('sell')
            confidence_factors.append(0.6)

        # Calculate Final Recommendation
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        confidence = np.mean(confidence_factors)
        predictions['confidence'] = confidence * 100

        # Generate Specific Recommendations
        current_trend = 'bullish' if buy_signals > sell_signals else 'bearish'
        
        if current_trend == 'bullish':
            # Calculate entry points
            entry_point = max(predictions['support_levels'][-1] if predictions['support_levels'] else current_price * 0.95,
                            current_price * 0.98)
            target_price = min(predictions['resistance_levels'][0] if predictions['resistance_levels'] else current_price * 1.05,
                             current_price * 1.03)
            
            # Find nearest predicted date for target
            target_date = None
            if predictions['lstm'] is not None:
                for date, price in zip(predictions['lstm']['dates'], predictions['lstm']['prices']):
                    if price >= target_price:
                        target_date = date
                        break
            
            predictions['entry_points'] = [entry_point]
            predictions['exit_points'] = [target_price]
            if target_date:
                predictions['timeframes'] = [target_date]
            
            recommendations = []
            recommendations.append(f"STRONG BUY SIGNAL for {ticker}")
            recommendations.append(f"Recommended entry point: ${entry_point:.2f}")
            recommendations.append(f"Target price: ${target_price:.2f}")
            if target_date:
                recommendations.append(f"Expected target date: {target_date.strftime('%Y-%m-%d')}")
            recommendations.append(f"Confidence level: {predictions['confidence']:.1f}%")
            
            # Add support for the recommendation
            recommendations.append("\nSupporting factors:")
            if rsi < 40:
                recommendations.append(f"- RSI indicates oversold conditions ({rsi:.1f})")
            if current_macd > current_signal:
                recommendations.append("- MACD shows positive momentum")
            if bb_position < 0.5:
                recommendations.append("- Price is in favorable Bollinger Band range")
            
            # Add risk factors
            recommendations.append("\nRisk factors to consider:")
            if predictions['resistance_levels']:
                recommendations.append(f"- Major resistance at ${predictions['resistance_levels'][0]:.2f}")
            
            predictions['recommendation'] = "\n".join(recommendations)
            
        else:
            # Bearish trend logic
            exit_point = min(predictions['resistance_levels'][0] if predictions['resistance_levels'] else current_price * 1.05,
                           current_price * 1.02)
            
            recommendations = []
            recommendations.append(f"SELL/HOLD SIGNAL for {ticker}")
            recommendations.append(f"Current price near resistance at ${exit_point:.2f}")
            recommendations.append(f"Confidence level: {predictions['confidence']:.1f}%")
            
            if predictions['support_levels']:
                recommendations.append(f"\nNext support level: ${predictions['support_levels'][-1]:.2f}")
            
            predictions['recommendation'] = "\n".join(recommendations)

        return predictions

    except Exception as e:
        print(f"Error in advanced prediction: {e}")
        return None
    
    

def create_advanced_lstm(sequence_length=60, n_features=1):
    """
    Create an enhanced LSTM model with attention mechanism
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        Attention(),
        LSTM(100),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
def prepare_advanced_features(data, sequence_length=60):
    """
    Prepare advanced features for neural network models
    """
    try:
        # Create a copy and ensure datetime index
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        
        # Technical indicators
        data['RSI'] = talib.RSI(data['Close'])
        data['MACD'], _, _ = talib.MACD(data['Close'])
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
        
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Volume-based features
        data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA20']
        
        # Use .ffill() instead of .fillna(method='bfill')
        data = data.ffill()
        
        # Prepare feature columns
        feature_columns = ['Close', 'RSI', 'MACD', 'ADX', 'ATR', 'Returns', 
                          'Volatility', 'MA20', 'MA50', 'Volume_Ratio']
        
        # Drop rows with NaN values
        data_cleaned = data.dropna(subset=feature_columns)
        
        # Normalize features
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(data_cleaned[feature_columns])
        
        # Prepare sequences
        X, y = [], []
        for i in range(sequence_length, len(features_normalized)):
            X.append(features_normalized[i-sequence_length:i])
            y.append(features_normalized[i, 0])  # 0 index for Close price
            
        return np.array(X), np.array(y), scaler
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None, None, None

def predict_with_advanced_models(data, future_days=10):
    """
    Make predictions using enhanced LSTM and additional features
    """
    try:
        sequence_length = 60
        X, y, scaler = prepare_advanced_features(data, sequence_length)
        
        if X is None or len(X) < sequence_length:
            print(f"Not enough data for prediction. Need at least {sequence_length} data points.")
            return None, None, None
            
        # Ensure 3D input for LSTM
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data for training
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Create and train the model
        model = create_advanced_lstm(sequence_length=X.shape[1], n_features=X.shape[2])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train with early stopping
        model.fit(X_train, y_train,
                 epochs=50,
                 batch_size=32,
                 validation_data=(X_test, y_test),
                 callbacks=[early_stopping],
                 verbose=0)
        
        # Prepare input for future prediction
        last_sequence = X[-1:]
        
        # Make sequential predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(future_days):
            current_pred = model.predict(current_sequence, verbose=0)[0]
            predictions.append(current_pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = current_pred
        
        # Transform predictions back to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        denormalized_predictions = scaler.inverse_transform(
            np.hstack([predictions_array, np.zeros((len(predictions), scaler.n_features_in_ - 1))])
        )[:, 0]
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=future_days, 
                                   freq='B')  # Business days
        
        # Calculate prediction metrics
        confidence = calculate_prediction_confidence(model, X_test, y_test)
        trend_accuracy = calculate_trend_accuracy(y_test, model.predict(X_test))
        
        prediction_metrics = {
            'confidence': confidence,
            'trend_accuracy': trend_accuracy
        }
        
        return future_dates, denormalized_predictions, prediction_metrics
        
    except Exception as e:
        print(f"Error in advanced prediction: {e}")
        return None, None, None

def calculate_prediction_confidence(model, X_test, y_test):
    """
    Calculate confidence score for predictions
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    confidence = np.exp(-mse) * 100  # Transform MSE to 0-100 scale
    return min(max(confidence, 0), 100)  # Clip to 0-100 range

def calculate_trend_accuracy(y_true, y_pred):
    """
    Calculate accuracy of predicted price movement directions
    """
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction) * 100
def calculate_prediction_range(predictions, confidence):
    """Add upper and lower bounds to predictions"""
    margin = (1 - (confidence/100)) * np.std(predictions)
    upper_bound = predictions + margin
    lower_bound = predictions - margin
    return upper_bound, lower_bound

def incorporate_market_sentiment(predictions, sentiment_score):
    """Adjust predictions based on market sentiment"""
    sentiment_factor = (sentiment_score - 50) / 100  # -0.5 to 0.5
    return predictions * (1 + sentiment_factor * 0.1)

def predict_volatility(data):
    """Predict future volatility"""
    returns = np.log(data['Close']/data['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    return volatility
# Define callback to update graph and predictions
@app.callback(
        [Output('stock-graph', 'figure'),
     Output('prediction-output', 'children'),
     Output('news-sentiment-container', 'style'),
     Output('sentiment-score', 'children'),
     Output('sentiment-score', 'style'),
     Output('pros-list', 'children'),
     Output('cons-list', 'children'),
     Output('risks-list', 'children'),
     Output('news-sources-list', 'children')],
    [Input('fetch-button', 'n_clicks'),
     Input('lstm-button', 'n_clicks'),
     Input('lstm-advanced-button', 'n_clicks'),
     Input('arima-button', 'n_clicks'),
     Input('ets-button', 'n_clicks'),
     Input('news-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('timeframe-dropdown', 'value'),
     State('news-sentiment-container', 'style')]
)
def update_dashboard(fetch_n_clicks, lstm_n_clicks, lstm_advanced_n_clicks, arima_n_clicks, ets_n_clicks, news_n_clicks, 
                     ticker, timeframe, news_container_style):
    # Initialize default return values
    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data to display")
    
    # Initialize all variables that will be returned
    score_style = {
        'width': '60px', 
        'height': '60px', 
        'borderRadius': '50%', 
        'display': 'flex', 
        'justifyContent': 'center', 
        'alignItems': 'center',
        'fontWeight': 'bold',
        'fontSize': '20px',
        'color': 'white',
        'backgroundColor': '#888'
    }
    news_visible = {'display': 'none'}
    sentiment_score = "--"
    pros = []
    cons = []
    risks = []
    news_sources = []
    future_dates = []
    predictions = []
    metrics = []
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check for valid ticker input
    if not ticker:
        return (empty_fig, "Please enter a stock ticker.", 
                {'display': 'none'}, "--", score_style, [], [], [], [])

    # Fetch the stock data
    data = fetch_stock_data(ticker, period=timeframe)
    if data.empty:
        return (empty_fig, f"No data found for ticker {ticker}.",
                {'display': 'none'}, "--", score_style, [], [], [], [])

    # Update news visibility based on container style
    if news_container_style != {'display': 'none'}:
        news_visible = {'display': 'block', 'marginTop': '30px'}
    sentiment_color = "#888"
    
    # Add technical indicators
    data_with_indicators = add_indicators(data)

    # Plot stock data
    fig = go.Figure()

    # Plot closing prices as a line
    fig.add_trace(go.Scatter(
        x=data_with_indicators.index, 
        y=data_with_indicators['Close'],
        mode='lines', 
        name="Close Price",
        line=dict(color='royalblue', width=2)
    ))

    # Generate stock price prediction based on indicators
    prediction_text = "Technical Analysis:\n\n"
    try:
        # Calculate RSI
        current_rsi = data_with_indicators['RSI'].iloc[-1]
        
        # Calculate MACD
        current_macd = data_with_indicators['MACD'].iloc[-1]
        current_signal = data_with_indicators['MACD_signal'].iloc[-1]
        
        # Determine recommendation based on RSI
        if current_rsi > 70:
            rsi_prediction = f"RSI: {current_rsi:.2f} - Overbought, consider selling."
        elif current_rsi < 30:
            rsi_prediction = f"RSI: {current_rsi:.2f} - Oversold, consider buying."
        else:
            rsi_prediction = f"RSI: {current_rsi:.2f} - Neutral, hold your position."
        
        # Determine recommendation based on MACD
        if current_macd > current_signal:
            macd_prediction = f"MACD: {current_macd:.2f}, Signal: {current_signal:.2f} - Bullish, consider buying."
        else:
            macd_prediction = f"MACD: {current_macd:.2f}, Signal: {current_signal:.2f} - Bearish, consider selling."
        
        prediction_text += f"{rsi_prediction}\n{macd_prediction}\n"
    except Exception as e:
        prediction_text += f"Error calculating technical indicators: {str(e)}\n"
    
    # Get buy/sell signals
    signals = get_buy_sell_signals(data_with_indicators)
    if signals:
        prediction_text += "\nTrading Signals:\n" + "\n".join(signals)

    # LSTM prediction if button is clicked
    if button_id == 'lstm-button':
        pred_dates, pred_prices = predict_with_lstm(data)
        if pred_dates is not None and pred_prices is not None:
            # Add the LSTM prediction trace
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines',
                name='LSTM Prediction',
                line=dict(color='blue', dash='dash')
            ))
            
            # Add marker for last actual price
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[data['Close'].iloc[-1]],
                mode='markers',
                marker=dict(color='blue', size=8),
                name='Last Actual Price'
            ))
            
            prediction_text += "\n\nLSTM Prediction added to chart."
        else:
            prediction_text += "\n\nLSTM prediction failed. Please try with more data."
    if button_id == 'lstm-advanced-button':
    # Get advanced LSTM predictions
        future_dates, predictions, metrics = predict_with_advanced_models(data)
        if future_dates is not None and predictions is not None and metrics is not None:
            # Calculate prediction ranges
            if isinstance(metrics, dict) and 'confidence' in metrics:
                upper_bound, lower_bound = calculate_prediction_range(predictions, metrics['confidence'])
            else:
                upper_bound = predictions * 1.1
                lower_bound = predictions * 0.9
            # Calculate volatility
            volatility = predict_volatility(data)
        
            # Add the main prediction trace
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Advanced LSTM Prediction',
                line=dict(color='purple', dash='dash')
            ))
        
            # Add prediction range
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(128, 0, 128, 0.2)', dash='dot')
            ))
        
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                name='Lower Bound',
                fill='tonexty',  # Fill area between bounds
                fillcolor='rgba(128, 0, 128, 0.1)',
                line=dict(color='rgba(128, 0, 128, 0.2)', dash='dot')
            ))
        
            # Create prediction text with enhanced metrics
            prediction_text = f"""
            Advanced LSTM Prediction Metrics:
            Confidence Score: {metrics['confidence']:.1f}%
            Trend Accuracy: {metrics['trend_accuracy']:.1f}%
            Predicted Volatility: {volatility:.2f}%
        
            Prediction Summary:
            Current Price: ${data['Close'].iloc[-1]:.2f}
            Predicted Price: ${predictions[-1]:.2f}
            Predicted Range: ${lower_bound[-1]:.2f} - ${upper_bound[-1]:.2f}
            """
        else:
            prediction_text = "Could not generate advanced LSTM predictions."
            upper_bound = None
            lower_bound = None
            volatility = None
    # ARIMA prediction if button is clicked
    if button_id == 'arima-button':
        forecasted_prices = simulate_stock(data)
        if not forecasted_prices.empty:
            fig.add_trace(go.Scatter(
                x=forecasted_prices.index, 
                y=forecasted_prices,
                mode='lines', 
                name="ARIMA Prediction",
                line=dict(color='red', width=2, dash='dash')
            ))
            prediction_text += "\n\nARIMA model prediction added to the chart."
        else:
            prediction_text += "\n\nARIMA simulation failed. Please try again with more data."

    # ETS prediction if button is clicked
    if button_id == 'ets-button':
        forecasted_prices = simulate_stock_ets(data)
        if not forecasted_prices.empty:
            fig.add_trace(go.Scatter(
                x=forecasted_prices.index, 
                y=forecasted_prices,
                mode='lines', 
                name="ETS Prediction",
                line=dict(color='purple', width=2, dash='dash')
            ))
            prediction_text += "\n\nETS model prediction added to the chart."
        else:
            prediction_text += "\n\nETS simulation failed. Please try again with more data."
    
    # News Analysis if button is clicked
    if button_id == 'news-button':
        news_visible = {'display': 'block', 'marginTop': '30px'}
        
        # Fetch news and analyze sentiment
        news_articles = fetch_stock_news(ticker)
        sentiment_analysis = analyze_news_sentiment(news_articles)
        
        # Update sentiment score and color
        sentiment_score = sentiment_analysis['score']
        
        # Determine color based on score (red to green gradient)
        if sentiment_score < 40:
            sentiment_color = f"rgb(255, {int(sentiment_score * 2.5)}, 0)"  # Red to yellow
        elif sentiment_score < 60:
            sentiment_color = f"rgb({int((80 - sentiment_score) * 6.375)}, 255, 0)"  # Yellow to green
        else:
            sentiment_color = f"rgb(0, 255, {int((sentiment_score - 60) * 2.5)})"  # Green getting brighter
            
        # Update sentiment score style
        score_style = {
            'width': '60px', 
            'height': '60px', 
            'borderRadius': '50%', 
            'display': 'flex', 
            'justifyContent': 'center', 
            'alignItems': 'center',
            'fontWeight': 'bold',
            'fontSize': '20px',
            'color': 'white',
            'backgroundColor': sentiment_color
        }
        
        # Update pros, cons, and risks lists
        pros = [html.Li(pro, style={'marginBottom': '8px'}) for pro in sentiment_analysis['pros']]
        cons = [html.Li(con, style={'marginBottom': '8px'}) for con in sentiment_analysis['cons']]
        risks = [html.Li(risk, style={'marginBottom': '8px'}) for risk in sentiment_analysis['risks']]
        
        # Create news sources list
        news_sources = [
            html.Div([
                html.A(
                    article.get('title', 'No title available'),
                    href=article.get('link', '#'),
                    target="_blank",
                    style={'color': '#2196F3', 'textDecoration': 'none', 'marginBottom': '5px', 'display': 'block'}
                ),
                html.Small(
                    article.get('published', 'Date not available'),
                    style={'color': '#666', 'display': 'block', 'marginBottom': '10px'}
                )
            ]) for article in news_articles[:5]  # Show only the 5 most recent news items
        ]

    # Add moving averages to chart
    fig.add_trace(go.Scatter(
        x=data_with_indicators.index, 
        y=data_with_indicators['SMA_50'],
        mode='lines', 
        name="SMA 50",
        line=dict(color='orange', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=data_with_indicators.index, 
        y=data_with_indicators['EMA_10'],
        mode='lines', 
        name="EMA 10",
        line=dict(color='magenta', width=1.5)
    ))

    # Update layout
    fig.update_layout(
        title=f"Stock Price Analysis for {ticker.upper()}",
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Generate advanced predictions
    advanced_predictions = generate_advanced_prediction(data_with_indicators, ticker)
    
    if advanced_predictions:
        recommendation_text = html.H4(advanced_predictions['recommendation'].split('\n')[0],  # Main recommendation
                                    style={'color': '#1a73e8', 'marginBottom': '15px'})
        
        # Format entry/exit points
        entry_exit_html = []
        if advanced_predictions['entry_points']:
            entry_exit_html.append(html.Div([
                html.Strong("Entry Point: "),
                f"${advanced_predictions['entry_points'][0]:.2f}"
            ], style={'marginBottom': '10px'}))
        if advanced_predictions['exit_points']:
            entry_exit_html.append(html.Div([
                html.Strong("Target Price: "),
                f"${advanced_predictions['exit_points'][0]:.2f}"
            ], style={'marginBottom': '10px'}))
        if advanced_predictions['timeframes']:
            entry_exit_html.append(html.Div([
                html.Strong("Expected Target Date: "),
                advanced_predictions['timeframes'][0].strftime('%Y-%m-%d')
            ], style={'marginBottom': '10px'}))
            
        # Format confidence indicator
        confidence_html = html.Div([
            html.Strong("Confidence Level: "),
            html.Span(f"{advanced_predictions['confidence']:.1f}%",
                     style={'color': '#00c853' if advanced_predictions['confidence'] > 70 else '#ff9800'})
        ], style={'marginBottom': '15px'})
        
        # Format supporting factors
        supporting_factors = [factor for factor in advanced_predictions['recommendation'].split('\n')
                            if factor.startswith('- ') and 'Risk' not in factor]
        supporting_html = html.Div([
            html.H5("Supporting Factors:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.Ul([html.Li(factor[2:]) for factor in supporting_factors],
                   style={'paddingLeft': '20px'})
        ]) if supporting_factors else ""
        
        # Format risk factors
        risk_factors = [factor for factor in advanced_predictions['recommendation'].split('\n')
                       if factor.startswith('- ') and 'Risk' in advanced_predictions['recommendation']]
        risk_html = html.Div([
            html.H5("Risk Factors:", style={'color': '#e53935', 'marginBottom': '10px'}),
            html.Ul([html.Li(factor[2:]) for factor in risk_factors],
                   style={'paddingLeft': '20px'})
        ]) if risk_factors else ""
    else:
        recommendation_text = "Unable to generate advanced predictions"
        entry_exit_html = ""
        confidence_html = ""
        supporting_html = ""
        risk_html = ""

    # Generate advanced predictions
    advanced_predictions = generate_advanced_prediction(data_with_indicators, ticker)
    
    if advanced_predictions:
        prediction_components = []
        
        # Add main recommendation
        prediction_components.append(html.H4(
            advanced_predictions['recommendation'].split('\n')[0],
            style={'color': '#1a73e8', 'marginBottom': '15px', 'fontWeight': 'bold'}
        ))
        
        # Add entry/exit points
        if advanced_predictions['entry_points']:
            prediction_components.extend([
                html.Div([
                    html.Strong("Entry Point: "),
                    html.Span(
                        f"${advanced_predictions['entry_points'][0]:.2f}",
                        style={'color': '#00c853'}
                    )
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Strong("Target Price: "),
                    html.Span(
                        f"${advanced_predictions['exit_points'][0]:.2f}" if advanced_predictions['exit_points'] else "N/A",
                        style={'color': '#ff9800'}
                    )
                ], style={'marginBottom': '10px'})
            ])
        
        # Add target date if available
        if advanced_predictions['timeframes']:
            prediction_components.append(
                html.Div([
                    html.Strong("Expected Target Date: "),
                    html.Span(advanced_predictions['timeframes'][0].strftime('%Y-%m-%d'))
                ], style={'marginBottom': '15px'})
            )
        
        # Add confidence level
        prediction_components.append(
            html.Div([
                html.Strong("Confidence Level: "),
                html.Span(
                    f"{advanced_predictions['confidence']:.1f}%",
                    style={'color': '#00c853' if advanced_predictions['confidence'] > 70 else '#ff9800'}
                )
            ], style={'marginBottom': '20px'})
        )
        
        # Add supporting factors
        supporting_factors = [factor for factor in advanced_predictions['recommendation'].split('\n')
                            if factor.startswith('- ') and 'Risk' not in factor]
        if supporting_factors:
            prediction_components.extend([
                html.H5("Supporting Factors:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.Ul([html.Li(factor[2:]) for factor in supporting_factors],
                       style={'paddingLeft': '20px', 'marginBottom': '15px'})
            ])
        
        # Add risk factors
        risk_factors = [factor for factor in advanced_predictions['recommendation'].split('\n')
                       if factor.startswith('- ') and 'Risk' in advanced_predictions['recommendation']]
        if risk_factors:
            prediction_components.extend([
                html.H5("Risk Factors:", style={'color': '#e53935', 'marginBottom': '10px'}),
                html.Ul([html.Li(factor[2:]) for factor in risk_factors],
                       style={'paddingLeft': '20px'})
            ])
        
        prediction_text = html.Div(prediction_components)
    else:
        prediction_text = "Unable to generate predictions at this time."

    return (
        fig,                # stock graph
        prediction_text,    # prediction output (now includes all advanced analysis)
        news_visible,      # news container visibility
        sentiment_score,   # sentiment score
        score_style,       # sentiment score style
        pros,              # pros list
        cons,              # cons list
        risks,             # risks list
        news_sources       # news sources list
    )


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=PORT)

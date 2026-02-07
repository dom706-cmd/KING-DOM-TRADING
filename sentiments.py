"""
KING DOM TRADING SYSTEM - Pro Features: Sentiment & Options Data
"""
import requests
import pandas as pd
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, finnhub_key):
        self.finnhub_key = finnhub_key

    def get_news_sentiment(self, ticker, hours=24):
        """Fetch recent news and calculate sentiment score (-1 to +1)."""
        try:
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={(pd.Timestamp.now()-pd.Timedelta(hours=hours)).strftime('%Y-%m-%d')}&to={pd.Timestamp.now().strftime('%Y-%m-%d')}&token={self.finnhub_key}"
            news = requests.get(url).json()
            if not news or isinstance(news, dict):
                return 0.0
            headlines = ' '.join([item.get('headline', '') for item in news[:10]])  # Top 10 headlines
            polarity = TextBlob(headlines).sentiment.polarity
            return round(polarity, 3)
        except Exception as e:
            logger.warning(f"News sentiment failed for {ticker}: {e}")
            return 0.0

    def get_put_call_ratio(self, ticker):
        """Fetch the put/call ratio for a given stock using Finnhub."""
        try:
            # Finnhub offers basic options data. For aggregate PCR, you might use CBOE.
            url = f"https://finnhub.io/api/v1/stock/option-chain?symbol={ticker}&token={self.finnhub_key}"
            data = requests.get(url).json()
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                put_volume = df[df['type'] == 'put']['volume'].sum()
                call_volume = df[df['type'] == 'call']['volume'].sum()
                return round(put_volume / call_volume, 3) if call_volume > 0 else 1.0
        except Exception as e:
            logger.warning(f"Put/Call ratio failed for {ticker}: {e}")
        return 1.0  # Neutral default

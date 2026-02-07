"""
KING DOM TRADING SYSTEM - PRO SUMMER EDITION
PRO-SUMMER COMMERCIAL GRADE MARKET SCANNER
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from calculations import QuantitativeTradingSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketScanner:
    """PRO-SUMMER COMMERCIAL GRADE MARKET SCANNER"""
    
    def __init__(self, config_file='config.json'):
        self.logger = logging.getLogger(__name__)
        self.quant_system = QuantitativeTradingSystem()
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # API Keys (Sign up for FREE at these URLs)
        self.finnhub_key = self.config.get('finnhub_key', 'YOUR_KEY_HERE')  # https://finnhub.io/register
        self.polygon_key = self.config.get('polygon_key', 'YOUR_KEY_HERE')  # https://polygon.io/
        self.alpaca_key = self.config.get('alpaca_key', 'YOUR_KEY_HERE')    # https://alpaca.markets/
        
        # Trading parameters
        self.account_size = self.config.get('account_size', 10000)
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)
        self.min_volume = self.config.get('min_volume', 1000000)
        self.min_price = self.config.get('min_price', 1.00)
        
        # Sector ETFs for correlation
        self.sector_etfs = {
            'Technology': 'XLK',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Consumer': 'XLY',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication': 'XLC'
        }
        
        # Cache
        self.cache = {}
        self.cache_time = {}
        self.CACHE_DURATION = timedelta(minutes=5)
        
        self.logger.info("PRO-SUMMER Market Scanner Initialized!")
    
    def load_config(self, config_file):
        """Load configuration from file"""
        default_config = {
            'finnhub_key': 'YOUR_KEY_HERE',
            'polygon_key': 'YOUR_KEY_HERE',
            'alpaca_key': 'YOUR_KEY_HERE',
            'account_size': 10000,
            'max_risk_per_trade': 0.02,
            'min_volume': 1000000,
            'min_price': 1.00,
            'scan_workers': 10,
            'watchlist': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
                'JPM', 'BAC', 'WFC', 'GS', 'JNJ', 'PFE', 'WMT', 'XOM', 'CVX'
            ]
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found. Using defaults.")
            return default_config
    
    def get_from_cache(self, key):
        """Get item from cache if valid"""
        if key in self.cache:
            if datetime.now() - self.cache_time.get(key, datetime.now()) < self.CACHE_DURATION:
                return self.cache[key]
        return None
    
    def set_cache(self, key, value):
        """Set item in cache"""
        self.cache[key] = value
        self.cache_time[key] = datetime.now()
    
    # ========== REAL-TIME DATA ==========
    
    def get_real_time_quote(self, ticker):
        """Get real-time quote from Finnhub (FREE)"""
        cache_key = f"realtime_{ticker}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            result = {
                'price': data.get('c', 0),
                'change': data.get('d', 0),
                'percent_change': data.get('dp', 0),
                'high': data.get('h', 0),
                'low': data.get('l', 0),
                'open': data.get('o', 0),
                'previous_close': data.get('pc', 0),
                'timestamp': datetime.fromtimestamp(data.get('t')) if data.get('t') else None,
                'source': 'Finnhub'
            }
            
            self.set_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.warning(f"Real-time quote failed for {ticker}: {e}")
            # Fallback to yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                result = {
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'change': info.get('regularMarketChange', 0),
                    'percent_change': info.get('regularMarketChangePercent', 0),
                    'high': info.get('dayHigh', 0),
                    'low': info.get('dayLow', 0),
                    'open': info.get('open', 0),
                    'previous_close': info.get('previousClose', 0),
                    'timestamp': datetime.now(),
                    'source': 'YFinance (fallback)'
                }
                return result
            except:
                return None
    
    def get_order_book(self, ticker):
        """Get order book data from Polygon (FREE)"""
        cache_key = f"orderbook_{ticker}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={self.polygon_key}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if 'ticker' in data:
                ticker_data = data['ticker']
                
                # Get bid/ask data
                bid_price = ticker_data.get('bid', {}).get('price', 0) if isinstance(ticker_data.get('bid'), dict) else ticker_data.get('bid', 0)
                ask_price = ticker_data.get('ask', {}).get('price', 0) if isinstance(ticker_data.get('ask'), dict) else ticker_data.get('ask', 0)
                
                result = {
                    'bid': bid_price,
                    'ask': ask_price,
                    'bid_size': ticker_data.get('bidSize', 0),
                    'ask_size': ticker_data.get('askSize', 0),
                    'last_trade': ticker_data.get('lastTrade', {}),
                    'todays_change': ticker_data.get('todaysChange', 0),
                    'todays_change_percent': ticker_data.get('todaysChangePerc', 0),
                    'updated': ticker_data.get('updated', 0),
                    'source': 'Polygon'
                }
                
                self.set_cache(cache_key, result)
                return result
        
        except Exception as e:
            self.logger.warning(f"Order book failed for {ticker}: {e}")
        
        return None
    
    # ========== VOLUME PROFILE ==========
    
    def calculate_volume_profile(self, data, bins=20):
        """Calculate volume-at-price profile"""
        if data.empty or len(data) < 10:
            return None
        
        cache_key = f"volprofile_{data.index[-1]}_{bins}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Create price bins
            min_price = data['Low'].min()
            max_price = data['High'].max()
            
            if min_price == max_price:
                return None
            
            price_range = max_price - min_price
            bin_edges = np.linspace(min_price, max_price, bins + 1)
            
            volume_profile = {}
            
            for i in range(len(bin_edges) - 1):
                bin_low = bin_edges[i]
                bin_high = bin_edges[i + 1]
                bin_mid = (bin_low + bin_high) / 2
                
                # Sum volume for bars that overlap with this bin
                total_volume = 0
                
                for idx in range(len(data)):
                    bar_low = data['Low'].iloc[idx]
                    bar_high = data['High'].iloc[idx]
                    bar_volume = data['Volume'].iloc[idx]
                    
                    # Check if bar overlaps with bin
                    if bar_high >= bin_low and bar_low <= bin_high:
                        # Calculate overlap percentage
                        overlap_min = max(bar_low, bin_low)
                        overlap_max = min(bar_high, bin_high)
                        overlap_range = overlap_max - overlap_min
                        bar_range = bar_high - bar_low
                        
                        if bar_range > 0:
                            overlap_ratio = overlap_range / bar_range
                            total_volume += bar_volume * overlap_ratio
                        else:
                            total_volume += bar_volume * 0.5
                
                volume_profile[round(bin_mid, 2)] = total_volume
            
            # Find high volume nodes
            if volume_profile:
                sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
                high_volume_nodes = sorted_profile[:5]
                
                # Calculate Value Area (70% of volume)
                total_volume = sum(volume_profile.values())
                target_volume = total_volume * 0.7
                
                sorted_prices = sorted(volume_profile.items(), key=lambda x: x[0])
                cumulative = 0
                value_area_prices = []
                
                for price, vol in sorted_prices:
                    cumulative += vol
                    value_area_prices.append(price)
                    if cumulative >= target_volume:
                        break
                
                result = {
                    'profile': volume_profile,
                    'high_volume_nodes': high_volume_nodes,
                    'value_area_low': min(value_area_prices) if value_area_prices else 0,
                    'value_area_high': max(value_area_prices) if value_area_prices else 0,
                    'point_of_control': sorted_profile[0][0] if sorted_profile else 0,
                    'total_volume': total_volume,
                    'price_range': price_range
                }
                
                self.set_cache(cache_key, result)
                return result
        
        except Exception as e:
            self.logger.error(f"Volume profile error: {e}")
        
        return None
    def get_real_time_quote(self, ticker):
        """Get real-time quote from Finnhub (FREE)"""
        # ... [Previous get_real_time_quote code remains exactly as provided] ...
        except Exception as e:
            self.logger.warning(f"Real-time quote failed for {ticker}: {e}")
            # Fallback to yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                result = {
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'change': info.get('regularMarketChange', 0),
                    'percent_change': info.get('regularMarketChangePercent', 0),
                    'high': info.get('dayHigh', info.get('regularMarketDayHigh', 0)),
                    'low': info.get('dayLow', info.get('regularMarketDayLow', 0)),
                    'open': info.get('open', info.get('regularMarketOpen', 0)),
                    'previous_close': info.get('previousClose', 0),
                    'timestamp': datetime.now(),
                    'source': 'Yahoo Finance (fallback)'
                }
                
                self.set_cache(cache_key, result)
                return result
                
            except Exception as fallback_e:
                self.logger.error(f"Fallback also failed for {ticker}: {fallback_e}")
                return None

    # ========== MARKET DATA ENHANCEMENTS ==========

    def get_volume_profile_levels(self, ticker, lookback_days=30, price_bins=10):
        """Calculate key volume profile support/resistance levels"""
        cache_key = f"volprofile_{ticker}_{lookback_days}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            data = self.quant_system.get_stock_data(
                ticker, 
                period=f"{lookback_days}d", 
                interval="1d"
            )
            
            if data.empty:
                return {"poc": 0, "high_volume_nodes": [], "low_volume_nodes": []}
            
            # Use the quant system's volume profile calculator
            profile = self.quant_system.calculate_volume_profile(data, bins=price_bins)
            
            if not profile:
                return {"poc": 0, "high_volume_nodes": [], "low_volume_nodes": []}
            
            # Find Point of Control (POC) - price with highest volume
            sorted_profile = sorted(profile.items(), key=lambda x: x[1], reverse=True)
            poc_price = sorted_profile[0][0] if sorted_profile else 0
            
            # Identify high volume nodes (top 30% of volume)
            high_volume_nodes = []
            low_volume_nodes = []
            
            if sorted_profile:
                max_volume = max(profile.values())
                volume_threshold = max_volume * 0.3  # Top 30%
                
                for price, volume in sorted_profile:
                    if volume >= volume_threshold:
                        high_volume_nodes.append({"price": price, "volume": volume})
                    else:
                        low_volume_nodes.append({"price": price, "volume": volume})
            
            result = {
                "poc": round(poc_price, 2),
                "high_volume_nodes": high_volume_nodes[:5],  # Top 5 only
                "low_volume_nodes": low_volume_nodes[:3],
                "profile_range": {
                    "min": round(min(profile.keys()), 2),
                    "max": round(max(profile.keys()), 2)
                }
            }
            
            self.set_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Volume profile failed for {ticker}: {e}")
            return {"poc": 0, "high_volume_nodes": [], "low_volume_nodes": []}

    def get_put_call_ratio(self, ticker):
        """Get put/call ratio from Finnhub options data"""
        cache_key = f"pcr_{ticker}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://finnhub.io/api/v1/stock/option-chain?symbol={ticker}&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    options_data = data['data']
                    
                    # Calculate put/call ratio
                    put_volume = sum(item.get('volume', 0) for item in options_data 
                                   if item.get('type', '').upper() == 'PUT')
                    call_volume = sum(item.get('volume', 0) for item in options_data 
                                    if item.get('type', '').upper() == 'CALL')
                    
                    # Also get open interest
                    put_oi = sum(item.get('openInterest', 0) for item in options_data 
                               if item.get('type', '').upper() == 'PUT')
                    call_oi = sum(item.get('openInterest', 0) for item in options_data 
                                if item.get('type', '').upper() == 'CALL')
                    
                    pcr_volume = put_volume / call_volume if call_volume > 0 else 0
                    pcr_oi = put_oi / call_oi if call_oi > 0 else 0
                    
                    result = {
                        'volume_ratio': round(pcr_volume, 3),
                        'oi_ratio': round(pcr_oi, 3),
                        'put_volume': put_volume,
                        'call_volume': call_volume,
                        'put_oi': put_oi,
                        'call_oi': call_oi,
                        'total_volume': put_volume + call_volume,
                        'total_oi': put_oi + call_oi
                    }
                    
                    self.set_cache(cache_key, result)
                    return result
                    
        except Exception as e:
            self.logger.warning(f"Put/call ratio failed for {ticker}: {e}")
        
        # Return default/neutral values
        default_result = {
            'volume_ratio': 1.0,
            'oi_ratio': 1.0,
            'put_volume': 0,
            'call_volume': 0,
            'put_oi': 0,
            'call_oi': 0,
            'total_volume': 0,
            'total_oi': 0
        }
        
        self.set_cache(cache_key, default_result)
        return default_result

    def get_news_sentiment(self, ticker):
        """Get news sentiment from Finnhub and calculate sentiment score"""
        cache_key = f"sentiment_{ticker}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Get news from last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
            url += f"&from={start_date.strftime('%Y-%m-%d')}"
            url += f"&to={end_date.strftime('%Y-%m-%d')}"
            url += f"&token={self.finnhub_key}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                news_items = response.json()
                
                if news_items and isinstance(news_items, list):
                    # Simple sentiment analysis based on headlines
                    sentiment_scores = []
                    relevant_news = []
                    
                    for item in news_items[:10]:  # Analyze top 10 news
                        headline = item.get('headline', '')
                        summary = item.get('summary', '')
                        
                        if headline:
                            # Simple keyword-based sentiment
                            text = f"{headline} {summary}".lower()
                            
                            # Positive keywords
                            positive_words = ['up', 'gain', 'rise', 'surge', 'beat', 'win', 'approve', 
                                            'growth', 'profit', 'buy', 'bullish', 'strong']
                            
                            # Negative keywords  
                            negative_words = ['down', 'fall', 'drop', 'plunge', 'miss', 'lose', 'reject',
                                            'decline', 'loss', 'sell', 'bearish', 'weak', 'cut']
                            
                            pos_count = sum(1 for word in positive_words if word in text)
                            neg_count = sum(1 for word in negative_words if word in text)
                            
                            if pos_count + neg_count > 0:
                                score = (pos_count - neg_count) / (pos_count + neg_count)
                            else:
                                score = 0
                            
                            sentiment_scores.append(score)
                            relevant_news.append({
                                'headline': headline[:100] + '...' if len(headline) > 100 else headline,
                                'sentiment': round(score, 2),
                                'date': item.get('datetime', item.get('date', ''))
                            })
                    
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                    
                    result = {
                        'score': round(avg_sentiment, 3),
                        'trend': 'BULLISH' if avg_sentiment > 0.1 else 'BEARISH' if avg_sentiment < -0.1 else 'NEUTRAL',
                        'news_count': len(relevant_news),
                        'latest_news': relevant_news[:3]  # Top 3 news
                    }
                    
                    self.set_cache(cache_key, result)
                    return result
                    
        except Exception as e:
            self.logger.warning(f"News sentiment failed for {ticker}: {e}")
        
        # Return neutral default
        default_result = {
            'score': 0.0,
            'trend': 'NEUTRAL',
            'news_count': 0,
            'latest_news': []
        }
        
        self.set_cache(cache_key, default_result)
        return default_result

    # ========== TECHNICAL ANALYSIS ==========

    def analyze_multi_timeframe(self, ticker):
        """Analyze alignment across multiple timeframes"""
        timeframes = [
            {'period': '1d', 'interval': '1h', 'name': '1H'},
            {'period': '5d', 'interval': '15m', 'name': '15M'},
            {'period': '1mo', 'interval': '1d', 'name': '1D'},
            {'period': '3mo', 'interval': '1d', 'name': '3D'}
        ]
        
        analysis = {}
        
        for tf in timeframes:
            try:
                data = self.quant_system.get_stock_data(
                    ticker, 
                    period=tf['period'], 
                    interval=tf['interval']
                )
                
                if len(data) > 20:
                    # Calculate indicators for this timeframe
                    rsi = self.quant_system.calculate_rsi(data).iloc[-1]
                    macd, signal = self.quant_system.calculate_macd(data)
                    macd_value = macd.iloc[-1]
                    signal_value = signal.iloc[-1]
                    macd_signal = 'BULLISH' if macd_value > signal_value else 'BEARISH'
                    
                    # Get price trend
                    recent_prices = data['Close'].tail(5)
                    price_trend = 'UP' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'DOWN'
                    
                    analysis[tf['name']] = {
                        'rsi': round(rsi, 2),
                        'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
                        'macd': round(macd_value, 3),
                        'macd_signal': macd_signal,
                        'price_trend': price_trend,
                        'volume': int(data['Volume'].mean()),
                        'alignment_score': self._calculate_alignment_score(rsi, macd_value, price_trend)
                    }
                    
            except Exception as e:
                self.logger.warning(f"Timeframe {tf['name']} analysis failed for {ticker}: {e}")
                analysis[tf['name']] = {
                    'rsi': 50,
                    'rsi_signal': 'NEUTRAL',
                    'macd': 0,
                    'macd_signal': 'NEUTRAL',
                    'price_trend': 'FLAT',
                    'volume': 0,
                    'alignment_score': 0
                }
        
        # Calculate overall alignment
        alignment_scores = [tf['alignment_score'] for tf in analysis.values() if 'alignment_score' in tf]
        overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        analysis['overall'] = {
            'alignment_score': round(overall_alignment, 2),
            'alignment': 'STRONG BULLISH' if overall_alignment > 0.7 else 
                        'BULLISH' if overall_alignment > 0.3 else
                        'NEUTRAL' if overall_alignment > -0.3 else
                        'BEARISH' if overall_alignment > -0.7 else
                        'STRONG BEARISH',
            'timeframes_aligned': sum(1 for score in alignment_scores if abs(score) > 0.5),
            'total_timeframes': len(alignment_scores)
        }
        
        return analysis

    def _calculate_alignment_score(self, rsi, macd, price_trend):
        """Calculate alignment score between indicators (-1 to +1)"""
        # RSI component (-1 to +1)
        rsi_score = 0
        if rsi > 70:
            rsi_score = -1  # Overbought = bearish
        elif rsi < 30:
            rsi_score = 1   # Oversold = bullish
        else:
            rsi_score = (50 - rsi) / 20  # Scale between -1 and 1
        
        # MACD component (-1 to +1)
        macd_score = np.tanh(macd * 10)  # Scale MACD to reasonable range
        
        # Combine scores
        combined = (rsi_score + macd_score) / 2
        return np.clip(combined, -1, 1)

    def calculate_sector_correlation(self, ticker, lookback_days=30):
        """Calculate correlation with sector ETF"""
        cache_key = f"correlation_{ticker}_{lookback_days}"
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Get sector for this ticker
            sector = self._get_stock_sector(ticker)
            sector_etf = self.sector_etfs.get(sector, 'SPY')  # Default to SPY if sector not found
            
            # Get data for both ticker and sector ETF
            stock_data = self.quant_system.get_stock_data(
                ticker, period=f"{lookback_days}d", interval="1d"
            )
            sector_data = self.quant_system.get_stock_data(
                sector_etf, period=f"{lookback_days}d", interval="1d"
            )
            
            if stock_data.empty or sector_data.empty:
                return {"correlation": 0, "sector": sector, "etf": sector_etf, "trend": "UNKNOWN"}
            
            # Align dates
            aligned_data = pd.concat(
                [stock_data['Close'], sector_data['Close']], 
                axis=1, 
                join='inner'
            )
            aligned_data.columns = [ticker, sector_etf]
            
            # Calculate returns correlation
            returns = aligned_data.pct_change().dropna()
            correlation = returns.corr().iloc[0, 1]
            
            # Calculate beta (relative volatility)
            covariance = returns.cov().iloc[0, 1]
            sector_variance = returns[sector_etf].var()
            beta = covariance / sector_variance if sector_variance != 0 else 1
            
            # Performance relative to sector
            stock_return = (aligned_data[ticker].iloc[-1] / aligned_data[ticker].iloc[0] - 1) * 100
            sector_return = (aligned_data[sector_etf].iloc[-1] / aligned_data[sector_etf].iloc[0] - 1) * 100
            relative_performance = stock_return - sector_return
            
            result = {
                'correlation': round(correlation, 3),
                'beta': round(beta, 3),
                'sector': sector,
                'sector_etf': sector_etf,
                'stock_return_pct': round(stock_return, 2),
                'sector_return_pct': round(sector_return, 2),
                'relative_performance': round(relative_performance, 2),
                'outperformance': 'OUTPERFORMING' if relative_performance > 1 else 
                                'UNDERPERFORMING' if relative_performance < -1 else 
                                'IN_LINE'
            }
            
            self.set_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.warning(f"Sector correlation failed for {ticker}: {e}")
            return {"correlation": 0, "sector": "UNKNOWN", "etf": "SPY", "trend": "UNKNOWN"}

    def _get_stock_sector(self, ticker):
        """Determine which sector a stock belongs to"""
        # Simple mapping - in production, you'd use an API or database
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer', 'META': 'Technology', 'NVDA': 'Technology',
            'TSLA': 'Consumer', 'AMD': 'Technology', 'INTC': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'GS': 'Financials', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'WMT': 'Consumer', 'XOM': 'Energy', 'CVX': 'Energy'
        }
        
        return sector_map.get(ticker, 'Technology')  # Default to Technology

    # ========== PARALLEL SCANNING ==========

    def scan_single_stock(self, ticker):
        """Complete analysis for a single stock - runs in parallel"""
        try:
            start_time = time.time()
            
            # 1. Get real-time quote
            quote = self.get_real_time_quote(ticker)
            if not quote or quote['price'] < self.min_price:
                return None
            
            # 2. Get volume profile
            volume_profile = self.get_volume_profile_levels(ticker)
            
            # 3. Get put/call ratio
            options_data = self.get_put_call_ratio(ticker)
            
            # 4. Get news sentiment
            sentiment = self.get_news_sentiment(ticker)
            
            # 5. Multi-timeframe analysis
            timeframe_analysis = self.analyze_multi_timeframe(ticker)
            
            # 6. Sector correlation
            sector_correlation = self.calculate_sector_correlation(ticker)
            
            # 7. Technical indicators from quant system
            stock_data = self.quant_system.get_stock_data(ticker, period='1mo')
            if not stock_data.empty:
                rsi = self.quant_system.calculate_rsi(stock_data).iloc[-1]
                macd, signal = self.quant_system.calculate_macd(stock_data)
                atr = self.quant_system.calculate_atr(stock_data).iloc[-1]
                momentum = self.quant_system.calculate_momentum(stock_data)
                volatility = self.quant_system.calculate_volatility(stock_data)
                support, resistance = self.quant_system.calculate_support_resistance(stock_data)
            else:
                rsi = 50
                macd, signal = 0, 0
                atr = 0
                momentum = 0
                volatility = 0
                support, resistance = [], []
            
            # 8. Calculate risk metrics
            position_size = self._calculate_position_size(quote['price'], atr)
            
            # 9. Generate overall score
            overall_score = self._calculate_overall_score(
                rsi, momentum, sentiment['score'], 
                timeframe_analysis['overall']['alignment_score'],
                sector_correlation['relative_performance']
            )
            
            result = {
                'ticker': ticker,
                'quote': quote,
                'technical': {
                    'rsi': round(rsi, 2),
                    'macd': round(macd.iloc[-1] if hasattr(macd, 'iloc') else macd, 3),
                    'atr': round(atr, 2),
                    'momentum_pct': round(momentum, 2),
                    'volatility_pct': round(volatility, 2),
                    'support_levels': [round(s, 2) for s in support[:3]],
                    'resistance_levels': [round(r, 2) for r in resistance[:3]]
                },
                'volume_profile': volume_profile,
                'options': options_data,
                'sentiment': sentiment,
                'timeframe_analysis': timeframe_analysis,
                'sector': sector_correlation,
                'risk': {
                    'position_size': position_size,
                    'stop_loss': round(quote['price'] - (2 * atr), 2),
                    'take_profit': round(quote['price'] + (3 * atr), 2),
                    'risk_reward': 1.5
                },
                'overall_score': round(overall_score, 2),
                'signal': 'STRONG BUY' if overall_score > 0.7 else
                         'BUY' if overall_score > 0.3 else
                         'NEUTRAL' if overall_score > -0.3 else
                         'SELL' if overall_score > -0.7 else
                         'STRONG SELL',
                'scan_duration': round(time.time() - start_time, 2)
            }
            
            self.logger.info(f"Scanned {ticker}: {result['signal']} (Score: {result['overall_score']})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to scan {ticker}: {e}")
            return None

    def scan_watchlist(self, watchlist=None, max_workers=None):
        """Parallel scan of multiple stocks"""
        if watchlist is None:
            watchlist = self.config.get('watchlist', [])
        
        if max_workers is None:
            max_workers = self.config.get('scan_workers', 10)
        
        self.logger.info(f"Starting parallel scan of {len(watchlist)} stocks with {max_workers} workers")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scanning tasks
            future_to_ticker = {
                executor.submit(self.scan_single_stock, ticker): ticker 
                for ticker in watchlist
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Scan task failed for {ticker}: {e}")
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        scan_time = time.time() - start_time
        
        self.logger.info(f"Scan complete: {len(results)} stocks analyzed in {scan_time:.1f} seconds")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'scan_duration': round(scan_time, 2),
            'stocks_scanned': len(watchlist),
            'stocks_analyzed': len(results),
            'top_picks': results[:5],  # Top 5 picks
            'all_results': results,
            'market_summary': self._generate_market_summary(results)
        }

    def _calculate_position_size(self, price, atr):
        """Calculate position size based on risk management"""
        if price <= 0 or atr <= 0:
            return 0
        
        # Risk per trade = account_size * max_risk_per_trade
        risk_amount = self.account_size * self.max_risk_per_trade
        
        # Stop loss based on 2x ATR
        stop_distance = 2 * atr
        
        # Position size = risk_amount / stop_distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        # Convert to number of shares
        shares = int(position_size / price) if price > 0 else 0
        
        # Don't risk more than 5% of account on any trade
        max_shares = int((self.account_size * 0.05) / price)
        
        return min(shares, max_shares)

    def _calculate_overall_score(self, rsi, momentum, sentiment, alignment, relative_perf):
        """Calculate overall score from multiple factors (-1 to +1)"""
        # Weight the components
        weights = {
            'technical': 0.35,      # RSI and momentum
            'sentiment': 0.20,      # News sentiment
            'alignment': 0.25,      # Multi-timeframe alignment
            'relative': 0.20        # Relative sector performance
        }
        
        # Normalize RSI to -1 to +1 scale
        rsi_score = (50 - rsi) / 20  # RSI 30=1, 50=0, 70=-1
        rsi_score = np.clip(rsi_score, -1, 1)
        
        # Normalize momentum (-1 to +1)
        momentum_score = np.tanh(momentum / 10)
        
        # Technical score combines RSI and momentum
        technical_score = (rsi_score + momentum_score) / 2
        
        # Combine all scores
        overall = (
            technical_score * weights['technical'] +
            sentiment * weights['sentiment'] +
            alignment * weights['alignment'] +
            np.tanh(relative_perf / 20) * weights['relative']
        )
        
        return np.clip(overall, -1, 1)

    def _generate_market_summary(self, results):
        """Generate overall market summary from scan results"""
        if not results:
            return {"market_trend": "NEUTRAL", "strength": 0}
        
        # Calculate average scores
        avg_score = np.mean([r['overall_score'] for r in results])
        avg_sentiment = np.mean([r['sentiment']['score'] for r in results])
        
        # Count signals
        signal_counts = {
            'STRONG_BUY': sum(1 for r in results if r['overall_score'] > 0.7),
            'BUY': sum(1 for r in results if 0.3 < r['overall_score'] <= 0.7),
            'NEUTRAL': sum(1 for r in results if -0.3 <= r['overall_score'] <= 0.3),
            'SELL': sum(1 for r in results if -0.7 <= r['overall_score'] < -0.3),
            'STRONG_SELL': sum(1 for r in results if r['overall_score'] < -0.7)
        }
        
        # Determine market trend
        if avg_score > 0.3:
            market_trend = "BULLISH"
            strength = min(avg_score * 100, 100)
        elif avg_score < -0.3:
            market_trend = "BEARISH"
            strength = min(abs(avg_score) * 100, 100)
        else:
            market_trend = "NEUTRAL"
            strength = 0
        
        return {
            "market_trend": market_trend,
            "market_strength": round(strength),
            "average_score": round(avg_score, 3),
            "average_sentiment": round(avg_sentiment, 3),
            "signal_distribution": signal_counts,
            "total_stocks": len(results)
        }

    # ========== PROFESSIONAL REPORT GENERATION ==========

    def generate_report(self, scan_results, output_format='console'):
        """Generate professional report in various formats"""
        if output_format == 'console':
            return self._generate_console_report(scan_results)
        elif output_format == 'json':
            return self._generate_json_report(scan_results)
        elif output_format == 'csv':
            return self._generate_csv_report(scan_results)
        else:
            self.logger.warning(f"Unknown format {output_format}, using console")
            return self._generate_console_report(scan_results)

    def _generate_console_report(self, scan_results):
        """Generate professional console report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("KING DOM TRADING - PRO SUMMER SCANNER REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Scan Duration: {scan_results['scan_duration']} seconds")
        report_lines.append(f"Stocks Analyzed: {scan_results['stocks_analyzed']}/{scan_results['stocks_scanned']}")
        report_lines.append("-" * 80)
        
        # Market Summary
        market = scan_results['market_summary']
        report_lines.append("MARKET SUMMARY:")
        report_lines.append(f"  Overall Trend: {market['market_trend']} (Strength: {market['market_strength']}%)")
        report_lines.append(f"  Average Score: {market['average_score']}")
        report_lines.append(f"  Bullish Signals: {market['signal_distribution']['STRONG_BUY'] + market['signal_distribution']['BUY']}")
        report_lines.append(f"  Bearish Signals: {market['signal_distribution']['STRONG_SELL'] + market['signal_distribution']['SELL']}")
        report_lines.append("-" * 80)
        
        # Top Picks
        report_lines.append("TOP 5 PICKS:")
        for i, stock in enumerate(scan_results['top_picks'], 1):
            report_lines.append(f"{i}. {stock['ticker']}:")
            report_lines.append(f"   Price: ${stock['quote']['price']:.2f} ({stock['quote']['percent_change']:.2f}%)")
            report_lines.append(f"   Signal: {stock['signal']} (Score: {stock['overall_score']:.2f})")
            report_lines.append(f"   RSI: {stock['technical']['rsi']} | MACD: {stock['technical']['macd']:.3f}")
            report_lines.append(f"   Sentiment: {stock['sentiment']['trend']} ({stock['sentiment']['score']:.3f})")
            report_lines.append(f"   PCR: {stock['options']['volume_ratio']:.3f} | Sector: {stock['sector']['sector']}")
            report_lines.append(f"   Position Size: {stock['risk']['position_size']} shares | Stop: ${stock['risk']['stop_loss']:.2f}")
            report_lines.append("")
        
        # Volume Profile Analysis for Top Pick
        if scan_results['top_picks']:
            top = scan_results['top_picks'][0]
            report_lines.append("-" * 80)
            report_lines.append(f"VOLUME PROFILE ANALYSIS - {top['ticker']}:")
            report_lines.append(f"  Point of Control: ${top['volume_profile']['poc']:.2f}")
            report_lines.append(f"  Profile Range: ${top['volume_profile']['profile_range']['min']:.2f} - ${top['volume_profile']['profile_range']['max']:.2f}")
            report_lines.append(f"  Key Support/Resistance Levels:")
            
            for level in top['volume_profile']['high_volume_nodes'][:3]:
                report_lines.append(f"    ${level['price']:.2f} (Volume: {level['volume']:,})")
        
        # Multi-Timeframe Alignment for Top Pick
        if scan_results['top_picks']:
            top = scan_results['top_picks'][0]
            report_lines.append("-" * 80)
            report_lines.append(f"MULTI-TIMEFRAME ALIGNMENT - {top['ticker']}:")
            for tf_name, tf_data in top['timeframe_analysis'].items():
                if tf_name != 'overall':
                    report_lines.append(f"  {tf_name}: RSI {tf_data['rsi']} ({tf_data['rsi_signal']}) | "
                                      f"MACD {tf_data['macd']:.3f} | Trend: {tf_data['price_trend']}")
            report_lines.append(f"  Overall Alignment: {top['timeframe_analysis']['overall']['alignment']}")
        
        # Risk Disclaimer
        report_lines.append("=" * 80)
        report_lines.append("RISK DISCLAIMER:")
        report_lines.append("This report is for EDUCATIONAL and RESEARCH purposes only.")
        report_lines.append("Not financial advice. Trading involves substantial risk of loss.")
        report_lines.append("Past performance is not indicative of future results.")
        report_lines.append("You are solely responsible for your own investment decisions.")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

    def _generate_json_report(self, scan_results):
        """Generate JSON report for programmatic use"""
        import json
        
        # Add metadata
        full_report = {
            'metadata': {
                'system': 'KING DOM TRADING - PRO SUMMER SCANNER',
                'version': '1.0.0',
                'generated': datetime.now().isoformat(),
                'account_size': self.account_size,
                'max_risk_per_trade': self.max_risk_per_trade
            },
            'risk_disclaimer': "This report is for EDUCATIONAL and RESEARCH purposes only. Not financial advice. Trading involves substantial risk of loss.",
            'scan_results': scan_results
        }
        
        return json.dumps(full_report, indent=2, default=str)

    def _generate_csv_report(self, scan_results):
        """Generate CSV report for spreadsheet analysis"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header row
        writer.writerow([
            'Ticker', 'Price', 'Change%', 'Signal', 'Score',
            'RSI', 'MACD', 'Sentiment', 'PCR', 'Sector',
            'Sector Perf%', 'Alignment', 'Position Size', 'Stop Loss'
        ])
        
        # Data rows
        for stock in scan_results['all_results']:
            writer.writerow([
                stock['ticker'],
                f"${stock['quote']['price']:.2f}",
                f"{stock['quote']['percent_change']:.2f}%",
                stock['signal'],
                f"{stock['overall_score']:.2f}",
                f"{stock['technical']['rsi']:.1f}",
                f"{stock['technical']['macd']:.3f}",
                stock['sentiment']['trend'],
                f"{stock['options']['volume_ratio']:.3f}",
                stock['sector']['sector'],
                f"{stock['sector']['relative_performance']:.1f}%",
                stock['timeframe_analysis']['overall']['alignment'],
                stock['risk']['position_size'],
                f"${stock['risk']['stop_loss']:.2f}"
            ])
        
        return output.getvalue()

    # ========== MAIN EXECUTION ==========

    def run_full_scan(self, watchlist=None, output_format='console'):
        """Complete scan and report generation"""
        self.logger.info("Starting full market scan...")
        
        # Run the scan
        scan_results = self.scan_watchlist(watchlist)
        
        # Generate report
        report = self.generate_report(scan_results, output_format)
        
        self.logger.info("Full scan complete!")
        
        return report


def main():
    """Main execution function"""
    print("=" * 60)
    print("KING DOM TRADING SYSTEM - PRO SUMMER EDITION")
    print("=" * 60)
    print("Professional Market Scanner Initializing...")
    
    # Initialize scanner
    scanner = MarketScanner()
    
    # Run scan
    report = scanner.run_full_scan(output_format='console')
    
    # Display report
    print("\n" + report)
    
    # Save to file
    with open('scanner_report.txt', 'w') as f:
        f.write(report)
    
    print("\nReport saved to 'scanner_report.txt'")
    print("=" * 60)


if __name__ == "__main__":
    main()

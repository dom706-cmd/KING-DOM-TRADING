"""
KING DOM TRADING SYSTEM - PRO SUMMER EDITION
QUANTITATIVE CALCULATIONS ENGINE - COMMERCIAL GRADE
25+ Advanced Indicators | Portfolio Optimization | ML Models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QuantitativeTradingSystem:
    """COMMERCIAL GRADE quantitative analysis engine - 25+ indicators"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        self.ml_models = {}
    
    def get_stock_data(self, ticker, period='3mo', interval='1d'):
        """Get stock data with intelligent caching"""
        cache_key = f"{ticker}_{period}_{interval}"
        
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return data.copy()
        
        try:
            stock = yf.Ticker(ticker)
            
            if interval == '1d':
                data = stock.history(period=period)
            else:
                data = stock.history(period=period, interval=interval)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate base features for ML
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close'] * 100
            data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open'] * 100
            
            # Volatility features
            data['Realized_Vol_5'] = data['Returns'].rolling(5).std() * np.sqrt(252) * 100
            data['Realized_Vol_20'] = data['Returns'].rolling(20).std() * np.sqrt(252) * 100
            
            self.cache[cache_key] = (datetime.now(), data.copy())
            return data.copy()
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    # ========== CORE INDICATORS (15+) ==========
    
    def calculate_rsi(self, data, period=14):
        """Advanced RSI with smoothing"""
        if len(data) < period:
            return pd.Series([50] * len(data), index=data.index)
        
        delta = data['Close'].diff()
        
        # Wilder's smoothing method (professional standard)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add divergence detection
        rsi_smoothed = rsi.ewm(span=5).mean()
        
        return rsi.fillna(50), rsi_smoothed
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """MACD with histogram and signal strength"""
        if len(data) < slow:
            return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)
        
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        # Calculate MACD strength
        macd_strength = (macd - macd.rolling(20).mean()) / macd.rolling(20).std()
        
        return macd, signal_line, histogram, macd_strength
    
    def calculate_atr(self, data, period=14):
        """Average True Range with volatility bands"""
        if len(data) < period:
            return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)
        
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()  # Wilder's smoothing
        
        # ATR bands for volatility
        atr_upper = atr * 1.5
        atr_lower = atr * 0.5
        
        return atr.fillna(tr.mean()), atr_upper, atr_lower
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Bollinger Bands with %B and Bandwidth"""
        if len(data) < period:
            middle = data['Close']
            upper = data['Close']
            lower = data['Close']
            bandwidth = pd.Series([0] * len(data), index=data.index)
            percent_b = pd.Series([0.5] * len(data), index=data.index)
        else:
            middle = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            # %B indicator
            percent_b = (data['Close'] - lower) / (upper - lower)
            
            # Bandwidth indicator
            bandwidth = (upper - lower) / middle * 100
        
        return upper, middle, lower, percent_b, bandwidth
    
    def calculate_ichimoku(self, data):
        """Ichimoku Cloud - professional trend system"""
        if len(data) < 52:
            return {}, {}, {}, {}, {}
        
        # Tenkan-sen (Conversion Line)
        period9_high = data['High'].rolling(window=9).max()
        period9_low = data['Low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = data['High'].rolling(window=26).max()
        period26_low = data['Low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = data['High'].rolling(window=52).max()
        period52_low = data['Low'].rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = data['Close'].shift(-26)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    def calculate_fibonacci_levels(self, data, lookback=60):
        """Fibonacci Retracement and Extension Levels"""
        if len(data) < lookback:
            return {}
        
        recent = data.tail(lookback)
        high = recent['High'].max()
        low = recent['Low'].min()
        diff = high - low
        
        levels = {
            '0%': low,
            '23.6%': low + diff * 0.236,
            '38.2%': low + diff * 0.382,
            '50%': low + diff * 0.5,
            '61.8%': low + diff * 0.618,
            '78.6%': low + diff * 0.786,
            '100%': high,
            '127.2%': high + diff * 0.272,
            '161.8%': high + diff * 0.618,
            '261.8%': high + diff * 1.618
        }
        
        return levels
    
    def calculate_vwap(self, data):
        """Volume Weighted Average Price (intraday)"""
        if data.empty:
            return pd.Series([], index=data.index)
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        return vwap
    
    def calculate_adx(self, data, period=14):
        """Average Directional Index - trend strength"""
        if len(data) < period * 2:
            return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # +DM and -DM
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the DMs and TR
        atr = tr.rolling(period).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=data.index).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=data.index).rolling(period).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    
    def calculate_stochastic(self, data, k_period=14, d_period=3):
        """Stochastic Oscillator with smoothing"""
        if len(data) < k_period:
            return pd.Series([50] * len(data), index=data.index), pd.Series([50] * len(data), index=data.index)
        
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k.fillna(50), d.fillna(50)
    
    def calculate_williams_r(self, data, period=14):
        """Williams %R - momentum oscillator"""
        if len(data) < period:
            return pd.Series([-50] * len(data), index=data.index)
        
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
        
        return williams_r.fillna(-50)
    
    def calculate_cci(self, data, period=20):
        """Commodity Channel Index"""
        if len(data) < period:
            return pd.Series([0] * len(data), index=data.index)
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci.fillna(0)
    
    def calculate_mfi(self, data, period=14):
        """Money Flow Index - volume-weighted RSI"""
        if len(data) < period:
            return pd.Series([50] * len(data), index=data.index)
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow, index=data.index).rolling(period).sum()
        negative_mf = pd.Series(negative_flow, index=data.index).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        
        return mfi.fillna(50)
    
    def calculate_obv(self, data, smooth_period=20):
        """On-Balance Volume with signal line"""
        obv = [0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        obv_series = pd.Series(obv, index=data.index)
        obv_signal = obv_series.rolling(smooth_period).mean()
        
        return obv_series, obv_signal
    
    def calculate_volume_profile(self, data, bins=20):
        """Volume Profile with Value Area calculation"""
        if data.empty:
            return {}, 0, 0, 0
        
        # Create price bins
        min_price = data['Low'].min()
        max_price = data['High'].max()
        bin_edges = np.linspace(min_price, max_price, bins + 1)
        
        volume_profile = {}
        price_volume_pairs = []
        
        for i in range(len(bin_edges) - 1):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            price_level = (low + high) / 2
            
            # Sum volume for bars that touch this price range
            mask = (data['Low'] <= high) & (data['High'] >= low)
            volume = data.loc[mask, 'Volume'].sum()
            
            volume_profile[round(price_level, 2)] = volume
            price_volume_pairs.append((price_level, volume))
        
        # Calculate Point of Control (POC)
        if price_volume_pairs:
            poc_price = max(price_volume_pairs, key=lambda x: x[1])[0]
            
            # Calculate Value Area (70% of volume)
            sorted_pairs = sorted(price_volume_pairs, key=lambda x: x[1], reverse=True)
            total_volume = sum(v for _, v in price_volume_pairs)
            target_volume = total_volume * 0.7
            
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_pairs:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            
            return volume_profile, poc_price, value_area_high, value_area_low
        
        return volume_profile, 0, 0, 0
    
    # ========== ADVANCED STATISTICAL INDICATORS ==========
    
    def calculate_beta(self, stock_data, market_data='SPY'):
        """Calculate Beta relative to market"""
        if isinstance(market_data, str):
            market = self.get_stock_data(market_data, period='3mo')
            market_returns = market['Returns'].dropna()
        else:
            market_returns = market_data['Returns'].dropna()
        
        stock_returns = stock_data['Returns'].dropna()
        
        # Align dates
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned.columns = ['stock', 'market']
        
        if len(aligned) < 20:
            return 1.0, 0.0
        
        # Calculate beta and alpha
        covariance = aligned['stock'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        beta = covariance / market_variance if market_variance != 0 else 1.0
        
        # Calculate alpha (CAPM)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        market_return = aligned['market'].mean() * 252  # Annualized
        stock_return = aligned['stock'].mean() * 252  # Annualized
        
        alpha = stock_return - (risk_free_rate * 252 + beta * (market_return - risk_free_rate * 252))
        
        return beta, alpha
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe Ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sortino Ratio (downside deviation only)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 100.0  # Perfect returns
        
        downside_deviation = downside_returns.std()
        sortino = np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else 100.0
        
        return sortino
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        
        cumulative_returns = (1 + equity_curve).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk (VaR)"""
        if len(returns) < 20:
            return 0.0
        
        # Historical VaR
        var_hist = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        var_param = returns.mean() + stats.norm.ppf(1 - confidence_level) * returns.std()
        
        return min(var_hist, var_param)
    
    # ========== MACHINE LEARNING MODELS ==========
    
    def train_price_prediction_model(self, ticker, features=['Returns', 'Volume_Ratio', 'High_Low_Pct']):
        """Train Random Forest model for price prediction"""
        data = self.get_stock_data(ticker, period='1y')
        
        if len(data) < 100:
            return None, 0.0
        
        # Create features and target
        data['Target'] = data['Close'].shift(-5)  # Predict 5 days ahead
        data = data.dropna()
        
        X = data[features]
        y = data['Target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        score = model.score(X_test_scaled, y_test)
        
        # Store model
        self.ml_models[ticker] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'score': score
        }
        
        return model, score
    
    def predict_price(self, ticker, current_data):
        """Predict future price using trained ML model"""
        if ticker not in self.ml_models:
            return None, 0.0
        
        model_data = self.ml_models[ticker]
        features = model_data['features']
        
        # Prepare current features
        current_features = pd.DataFrame([current_data[features].iloc[-1]])
        scaled_features = model_data['scaler'].transform(current_features)
        
        # Make prediction
        prediction = model_data['model'].predict(scaled_features)[0]
        confidence = model_data['score']
        
        return prediction, confidence
    
    # ========== PORTFOLIO OPTIMIZATION ==========
    
    def optimize_portfolio(self, tickers, method='sharpe', risk_free_rate=0.02):
        """Modern Portfolio Theory optimization"""
        # Get returns for all tickers
        returns_data = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period='1y')
            if not data.empty:
                returns_data[ticker] = data['Returns'].dropna()
        
        if len(returns_data) < 2:
            return None, None
        
        # Align returns
        returns_df = pd.concat(returns_data, axis=1).dropna()
        returns_df.columns = list(returns_data.keys())
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Define objective function based on method
        def portfolio_performance(weights):
            returns = np.sum(weights * expected_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if method == 'sharpe':
                sharpe = (returns - risk_free_rate) / volatility if volatility != 0 else 0
                return -sharpe  # Minimize negative sharpe
            elif method == 'min_volatility':
                return volatility
            else:  # max_return
                return -returns
        
        # Constraints and bounds
        n_assets = len(expected_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
        
        # Initial guess
        init_guess = n_assets * [1. / n_assets]
        
        # Optimize
        optimized = optimize.minimize(portfolio_performance, 
                                    init_guess,
                                    method='SLSQP',
                                    bounds=bounds,
                                    constraints=constraints)
        
        if optimized.success:
            weights = optimized.x
            returns = np.sum(weights * expected_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (returns - risk_free_rate) / volatility if volatility != 0 else 0
            
            return weights, {
                'returns': returns,
                'volatility': volatility,
                'sharpe': sharpe,
                'weights': dict(zip(tickers, weights))
            }
        
        return None, None
    
    def calculate_efficient_frontier(self, tickers, points=50):
        """Calculate efficient frontier for portfolio visualization"""
        returns_data = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period='1y')
            if not data.empty:
                returns_data[ticker] = data['Returns'].dropna()
        
        if len(returns_data) < 2:
            return [], []
        
        # Align returns
        returns_df = pd.concat(returns_data, axis=1).dropna()
        returns_df.columns = list(returns_data.keys())
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        n_assets = len(expected_returns)
        
        # Generate random portfolios
        portfolios = []
        for _ in range(points * 10):  # Generate many, filter later
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            returns = np.sum(weights * expected_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            portfolios.append((volatility, returns, weights))
        
        # Find efficient frontier (max return for given risk)
        portfolios.sort(key=lambda x: x[0])  # Sort by volatility
        efficient_frontier = []
        current_max_return = -np.inf
        
        for vol, ret, _ in portfolios:
            if ret > current_max_return:
                efficient_frontier.append((vol, ret))
                current_max_return = ret
        
        return list(zip(*efficient_frontier)) if efficient_frontier else ([], [])
    
    # ========== RISK MANAGEMENT ==========
    
    def calculate_position_size(self, account_size, entry_price, stop_loss, risk_per_trade=0.02):
        """Calculate position size based on risk management"""
        risk_amount = account_size * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        shares = risk_amount / risk_per_share
        shares = int(shares)  # Whole shares
        
        # Check margin requirements (max 25% of account)
        position_value = shares * entry_price
        max_position = account_size * 0.25
        
        if position_value > max_position:
            shares = int(max_position / entry_price)
        
        return shares
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate Kelly Criterion optimal bet size"""
        if avg_loss >= 0:
            return 0.0
        
        win_prob = win_rate / 100.0 if win_rate > 1 else win_rate
        loss_prob = 1 - win_prob
        win_ratio = avg_win / abs(avg_loss)
        
        kelly = win_prob - (loss_prob / win_ratio)
        
        # Conservative Kelly (half-Kelly)
        return max(0, kelly / 2)
    
    # ========== SUPPORT/RESISTANCE ==========
    
    def calculate_support_resistance(self, data, lookback=20, method='pivot'):
        """Advanced support/resistance with multiple methods"""
        if len(data) < lookback:
            return [], [], []
        
        recent = data.tail(lookback)
        supports = []
        resistances = []
        pivot_points = {}
        
        if method == 'pivot':
            # Pivot Point method (professional)
            pivot = (recent['High'].iloc[-1] + recent['Low'].iloc[-1] + recent['Close'].iloc[-1]) / 3
            r1 = 2 * pivot - recent['Low'].iloc[-1]
            r2 = pivot + (recent['High'].iloc[-1] - recent['Low'].iloc[-1])
            s1 = 2 * pivot - recent['High'].iloc[-1]
            s2 = pivot - (recent['High'].iloc[-1] - recent['Low'].iloc[-1])
            
            pivot_points = {
                'pivot': pivot,
                'r1': r1, 'r2': r2,
                's1': s1, 's2': s2
            }
            
            resistances = [r1, r2]
            supports = [s1, s2]
        
        elif method == 'fractal':
            # Fractal method (Williams)
            for i in range(2, len(recent)-2):
                # Resistance fractal
                if (recent['High'].iloc[i] > recent['High'].iloc[i-2] and
                    recent['High'].iloc[i] > recent['High'].iloc[i-1] and
                    recent['High'].iloc[i] > recent['High'].iloc[i+1] and
                    recent['High'].iloc[i] > recent['High'].iloc[i+2]):
                    resistances.append(recent['High'].iloc[i])
                
                # Support fractal
                if (recent['Low'].iloc[i] < recent['Low'].iloc[i-2] and
                    recent['Low'].iloc[i] < recent['Low'].iloc[i-1] and
                    recent['Low'].iloc[i] < recent['Low'].iloc[i+1] and
                    recent['Low'].iloc[i] < recent['Low'].iloc[i+2]):
                    supports.append(recent['Low'].iloc[i])
        
        # Cluster similar levels
        supports_clustered = self._cluster_levels(supports, tolerance=0.01)
        resistances_clustered = self._cluster_levels(resistances, tolerance=0.01)
        
        return supports_clustered[:3], resistances_clustered[:3], pivot_points
    
    def _cluster_levels(self, levels, tolerance=0.02):
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels.sort()
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level <= current_cluster[-1] * (1 + tolerance):
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    # ========== MOMENTUM & VOLATILITY ==========
    
    def calculate_momentum(self, data, periods=[5, 10, 20, 50]):
        """Multi-period momentum analysis"""
        if len(data) < max(periods):
            return {f"{p}d": 0 for p in periods}
        
        momentum = {}
        for period in periods:
            if len(data) >= period:
                mom = ((data['Close'].iloc[-1] / data['Close'].iloc[-period]) - 1) * 100
                momentum[f"{period}d"] = round(mom, 2)
            else:
                momentum[f"{period}d"] = 0
        
        # Calculate momentum score (weighted average)
        weights = [0.1, 0.2, 0.3, 0.4]  # More weight to longer periods
        weighted_mom = sum(momentum[f"{p}d"] * w for p, w in zip(periods, weights[:len(periods)]))
        
        momentum['score'] = round(weighted_mom, 2)
        
        return momentum
    
    def calculate_volatility(self, data, periods=[5, 10, 20]):
        """Multi-period volatility analysis"""
        if len(data) < max(periods) + 1:
            return {f"{p}d": 0 for p in periods}
        
        volatility = {}
        for period in periods:
            returns = data['Close'].pct_change().tail(period)
            if len(returns) >= 2:
                vol = returns.std() * np.sqrt(252) * 100  # Annualized percentage
                volatility[f"{p}d"] = round(vol, 2)
            else:
                volatility[f"{p}d"] = 0
        
        # Average volatility
        volatility['avg'] = round(np.mean(list(volatility.values())), 2)
        
        return volatility


# ========== TESTING & DEMONSTRATION ==========

if __name__ == "__main__":
    print("=" * 70)
    print("KING DOM TRADING SYSTEM - PRO CALCULATIONS ENGINE")
    print("=" * 70)
    
    qts = QuantitativeTradingSystem()
    
    # Test with AAPL
    print("\nTesting with AAPL data...")
    test_data = qts.get_stock_data("AAPL", period="3mo", interval="1d")
    
    if not test_data.empty:
        print(f"✓ Retrieved {len(test_data)} days of data")
        print(f"✓ Features: {list(test_data.columns)}")
        
        # Test 15+ indicators
        print("\n" + "-" * 70)
        print("ADVANCED INDICATORS TEST:")
        print("-" * 70)
        
        # 1. RSI
        rsi, rsi_smooth = qts.calculate_rsi(test_data)
        print(f"1. RSI: {rsi.iloc[-1]:.2f} (Smoothed: {rsi_smooth.iloc[-1]:.2f})")
        
        # 2. MACD
        macd, signal, hist, strength = qts.calculate_macd(test_data)
        print(f"2. MACD: {macd.iloc[-1]:.3f}, Signal: {signal.iloc[-1]:.3f}, Histogram: {hist.iloc[-1]:.3f}")
        
        # 3. Bollinger Bands
        upper, middle, lower, percent_b, bandwidth = qts.calculate_bollinger_bands(test_data)
        print(f"3. Bollinger: Price at {percent_b.iloc[-1]:.1%} of band, Width: {bandwidth.iloc[-1]:.2f}%")
        
        # 4. Volume Profile
        profile, poc, va_high, va_low = qts.calculate_volume_profile(test_data)
        print(f"4. Volume Profile: POC=${poc:.2f}, Value Area: ${va_low:.2f}-${va_high:.2f}")
        
        # 5. Fibonacci Levels
        fib_levels = qts.calculate_fibonacci_levels(test_data)
        print(f"5. Fibonacci: 38.2% at ${fib_levels.get('38.2%', 0):.2f}, 61.8% at ${fib_levels.get('61.8%', 0):.2f}")
        
        # 6. Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = qts.calculate_ichimoku(test_data)
        if len(tenkan) > 0:
            print(f"6. Ichimoku: Tenkan ${tenkan.iloc[-1]:.2f}, Kijun ${kijun.iloc[-1]:.2f}")
        
        # 7. Advanced Statistics
        print("\n" + "-" * 70)
        print("ADVANCED STATISTICS:")
        print("-" * 70)
        
        beta, alpha = qts.calculate_beta(test_data)
        print(f"✓ Beta: {beta:.3f}, Alpha: {alpha:.3%}")
        
        sharpe = qts.calculate_sharpe_ratio(test_data['Returns'].dropna())
        print(f"✓ Sharpe Ratio: {sharpe:.3f}")
        
        sortino = qts.calculate_sortino_ratio(test_data['Returns'].dropna())
        print(f"✓ Sortino Ratio: {sortino:.3f}")
        
        var = qts.calculate_var(test_data['Returns'].dropna())
        print(f"✓ 95% VaR: {var:.3%}")
        
        # 8. Machine Learning
        print("\n" + "-" * 70)
        print("MACHINE LEARNING:")
        print("-" * 70)
        
        model, score = qts.train_price_prediction_model("AAPL")
        if model:
            print(f"✓ ML Model trained: R² = {score:.3f}")
            
            # Make prediction
            pred, confidence = qts.predict_price("AAPL", test_data)
            if pred:
                print(f"✓ 5-day price prediction: ${pred:.2f} (Confidence: {confidence:.1%})")
        
        # 9. Portfolio Optimization
        print("\n" + "-" * 70)
        print("PORTFOLIO OPTIMIZATION:")
        print("-" * 70)
        
        test_portfolio = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        weights, metrics = qts.optimize_portfolio(test_portfolio, method='sharpe')
        
        if weights is not None:
            print("✓ Optimized Portfolio Weights:")
            for ticker, weight in zip(test_portfolio, weights):
                if weight > 0.01:  # Only show >1% allocations
                    print(f"  {ticker}: {weight:.1%}")
            
            print(f"✓ Expected Return: {metrics['returns']:.1%}")
            print(f"✓ Expected Volatility: {metrics['volatility']:.1%}")
            print(f"✓ Sharpe Ratio: {metrics['sharpe']:.3f}")
        
        print("\n" + "=" * 70)
        print("PRO CALCULATIONS ENGINE TEST COMPLETE ✓")
        print("=" * 70)
        
    else:
        print("✗ Failed to retrieve test data")

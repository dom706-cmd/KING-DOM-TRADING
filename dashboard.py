"""
KING DOM TRADING SYSTEM - Professional Web Dashboard
File 6: dashboard.py
"""
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json
import threading
from market_scanner import MarketScanner
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

# Global scanner instance
scanner = None
last_scan_results = None
last_scan_time = None

# HTML Template with Tailwind CSS
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KING DOM TRADING - Pro Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .dashboard-card { transition: all 0.3s ease; }
        .dashboard-card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .signal-buy { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .signal-sell { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
        .signal-neutral { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <!-- Navigation -->
    <nav class="bg-gray-800 border-b border-gray-700">
        <div class="container mx-auto px-4 py-3">
            <div class="flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <i class="fas fa-crown text-yellow-400 text-2xl"></i>
                    <h1 class="text-2xl font-bold">KING DOM TRADING <span class="text-yellow-400">PRO DASHBOARD</span></h1>
                </div>
                <div class="flex items-center space-x-4">
                    <span id="last-update" class="text-gray-400">Last scan: {{ last_scan_time }}</span>
                    <button onclick="runScan()" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-semibold">
                        <i class="fas fa-sync-alt mr-2"></i>Run New Scan
                    </button>
                    <div class="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600"></div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Market Overview -->
    <div class="container mx-auto px-4 py-6">
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
            <div class="dashboard-card bg-gray-800 rounded-xl p-6 col-span-1 lg:col-span-2">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-chart-line text-green-400 mr-3"></i>Market Overview
                </h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div class="text-center p-4 rounded-lg bg-gray-700">
                        <div class="text-2xl font-bold text-green-400">{{ market_strength }}%</div>
                        <div class="text-sm text-gray-400">Market Strength</div>
                    </div>
                    <div class="text-center p-4 rounded-lg bg-gray-700">
                        <div class="text-2xl font-bold {{ 'text-green-400' if market_trend == 'BULLISH' else 'text-red-400' if market_trend == 'BEARISH' else 'text-gray-400' }}">
                            {{ market_trend }}
                        </div>
                        <div class="text-sm text-gray-400">Trend</div>
                    </div>
                    <div class="text-center p-4 rounded-lg bg-gray-700">
                        <div class="text-2xl font-bold text-blue-400">{{ stocks_analyzed }}</div>
                        <div class="text-sm text-gray-400">Stocks Scanned</div>
                    </div>
                    <div class="text-center p-4 rounded-lg bg-gray-700">
                        <div class="text-2xl font-bold text-yellow-400">{{ bullish_signals }}</div>
                        <div class="text-sm text-gray-400">Bullish Signals</div>
                    </div>
                </div>
            </div>

            <div class="dashboard-card bg-gray-800 rounded-xl p-6">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-balance-scale text-blue-400 mr-3"></i>Signal Distribution
                </h2>
                <canvas id="signalChart" height="150"></canvas>
            </div>

            <div class="dashboard-card bg-gray-800 rounded-xl p-6">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-bolt text-yellow-400 mr-3"></i>Top Performer
                </h2>
                {% if top_stock %}
                <div class="text-center">
                    <div class="text-3xl font-bold mb-2">{{ top_stock.ticker }}</div>
                    <div class="text-lg font-semibold {{ 'text-green-400' if top_stock.overall_score > 0 else 'text-red-400' }}">
                        Score: {{ "%.2f"|format(top_stock.overall_score) }}
                    </div>
                    <div class="text-sm text-gray-400 mt-2">{{ top_stock.signal }}</div>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Top 5 Stocks Table -->
        <div class="dashboard-card bg-gray-800 rounded-xl p-6 mb-8">
            <h2 class="text-xl font-bold mb-4 flex items-center">
                <i class="fas fa-trophy text-yellow-400 mr-3"></i>Top 5 Trading Opportunities
            </h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="text-left border-b border-gray-700">
                            <th class="pb-3">Ticker</th>
                            <th class="pb-3">Price</th>
                            <th class="pb-3">Signal</th>
                            <th class="pb-3">Score</th>
                            <th class="pb-3">RSI</th>
                            <th class="pb-3">Sentiment</th>
                            <th class="pb-3">PCR</th>
                            <th class="pb-3">Sector</th>
                            <th class="pb-3">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for stock in top_stocks %}
                        <tr class="border-b border-gray-700 hover:bg-gray-750">
                            <td class="py-4">
                                <div class="font-bold text-lg">{{ stock.ticker }}</div>
                            </td>
                            <td class="py-4">
                                <div class="font-semibold">${{ "%.2f"|format(stock.quote.price) }}</div>
                                <div class="text-sm {{ 'text-green-400' if stock.quote.percent_change > 0 else 'text-red-400' }}">
                                    {{ "%.2f"|format(stock.quote.percent_change) }}%
                                </div>
                            </td>
                            <td class="py-4">
                                <span class="px-3 py-1 rounded-full text-sm font-bold 
                                    {{ 'signal-buy' if stock.signal in ['STRONG BUY', 'BUY'] else 
                                       'signal-sell' if stock.signal in ['STRONG SELL', 'SELL'] else 
                                       'signal-neutral' }}">
                                    {{ stock.signal }}
                                </span>
                            </td>
                            <td class="py-4">
                                <div class="text-lg font-bold {{ 'text-green-400' if stock.overall_score > 0 else 'text-red-400' }}">
                                    {{ "%.2f"|format(stock.overall_score) }}
                                </div>
                            </td>
                            <td class="py-4">
                                <div class="{{ 'text-red-400' if stock.technical.rsi > 70 else 'text-green-400' if stock.technical.rsi < 30 else 'text-gray-400' }}">
                                    {{ "%.1f"|format(stock.technical.rsi) }}
                                </div>
                            </td>
                            <td class="py-4">
                                <div class="flex items-center">
                                    <i class="fas fa-{{ 'smile text-green-400' if stock.sentiment.score > 0.1 else 'frown text-red-400' if stock.sentiment.score < -0.1 else 'meh text-gray-400' }} mr-2"></i>
                                    {{ stock.sentiment.trend }}
                                </div>
                            </td>
                            <td class="py-4">
                                <div class="{{ 'text-red-400' if stock.options.volume_ratio > 1.2 else 'text-green-400' if stock.options.volume_ratio < 0.8 else 'text-gray-400' }}">
                                    {{ "%.3f"|format(stock.options.volume_ratio) }}
                                </div>
                            </td>
                            <td class="py-4">
                                <span class="px-2 py-1 bg-gray-700 rounded text-sm">{{ stock.sector.sector }}</span>
                            </td>
                            <td class="py-4">
                                <button onclick="showDetails('{{ stock.ticker }}')" class="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm mr-2">
                                    <i class="fas fa-chart-bar"></i>
                                </button>
                                <button onclick="runSingleScan('{{ stock.ticker }}')" class="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm">
                                    <i class="fas fa-sync"></i>
                                </button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Volume Profile Chart -->
            <div class="dashboard-card bg-gray-800 rounded-xl p-6">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-chart-bar text-purple-400 mr-3"></i>Volume Profile - {{ selected_ticker }}
                </h2>
                <div id="volumeProfileChart" class="h-80"></div>
            </div>

            <!-- Multi-Timeframe Alignment -->
            <div class="dashboard-card bg-gray-800 rounded-xl p-6">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-layer-group text-blue-400 mr-3"></i>Multi-Timeframe Alignment
                </h2>
                <div id="timeframeChart" class="h-80"></div>
            </div>
        </div>

        <!-- Detailed Analysis -->
        <div class="dashboard-card bg-gray-800 rounded-xl p-6 mb-8">
            <h2 class="text-xl font-bold mb-4 flex items-center">
                <i class="fas fa-search text-green-400 mr-3"></i>Detailed Analysis - <span id="detail-ticker" class="ml-2">{{ selected_ticker }}</span>
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                    <h3 class="font-bold mb-3 text-gray-300"><i class="fas fa-thermometer-half mr-2"></i>Technical Indicators</h3>
                    <div class="space-y-2">
                        <div class="flex justify-between"><span>RSI:</span><span id="detail-rsi" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>MACD:</span><span id="detail-macd" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>ATR:</span><span id="detail-atr" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Momentum:</span><span id="detail-momentum" class="font-semibold">-</span></div>
                    </div>
                </div>
                <div>
                    <h3 class="font-bold mb-3 text-gray-300"><i class="fas fa-newspaper mr-2"></i>Sentiment & Options</h3>
                    <div class="space-y-2">
                        <div class="flex justify-between"><span>News Sentiment:</span><span id="detail-sentiment" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Put/Call Ratio:</span><span id="detail-pcr" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Total Volume:</span><span id="detail-volume" class="font-semibold">-</span></div>
                    </div>
                </div>
                <div>
                    <h3 class="font-bold mb-3 text-gray-300"><i class="fas fa-shield-alt mr-2"></i>Risk Management</h3>
                    <div class="space-y-2">
                        <div class="flex justify-between"><span>Position Size:</span><span id="detail-position" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Stop Loss:</span><span id="detail-stop" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Take Profit:</span><span id="detail-take" class="font-semibold">-</span></div>
                        <div class="flex justify-between"><span>Risk/Reward:</span><span id="detail-rr" class="font-semibold">-</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Real-time Updates -->
        <div class="dashboard-card bg-gray-800 rounded-xl p-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold flex items-center">
                    <i class="fas fa-broadcast-tower text-red-400 mr-3 pulse"></i>Real-time Updates
                </h2>
                <div class="flex space-x-2">
                    <div class="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
                    <span class="text-sm">Live</span>
                </div>
            </div>
            <div id="live-updates" class="h-40 overflow-y-auto border border-gray-700 rounded p-3">
                <!-- Updates will appear here -->
                <div class="text-gray-400 text-center py-8">Waiting for scan results...</div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="bg-gray-800 border-t border-gray-700 py-6 mt-8">
        <div class="container mx-auto px-4 text-center text-gray-400">
            <div class="mb-4">
                <i class="fas fa-crown text-yellow-400 text-2xl"></i>
            </div>
            <p class="mb-2">KING DOM TRADING SYSTEM - PRO SUMMER EDITION</p>
            <p class="text-sm">This dashboard is for EDUCATIONAL and RESEARCH purposes only. Not financial advice.</p>
            <p class="text-xs mt-4">Trading involves substantial risk of loss. Past performance is not indicative of future results.</p>
        </div>
    </footer>

    <!-- JavaScript -->
    <script>
        let currentData = {{ scan_data|tojson }};
        
        // Initialize charts
        function initCharts() {
            // Signal Distribution Chart
            const signalCtx = document.getElementById('signalChart').getContext('2d');
            new Chart(signalCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'],
                    datasets: [{
                        data: [
                            currentData.market_summary.signal_distribution.STRONG_BUY,
                            currentData.market_summary.signal_distribution.BUY,
                            currentData.market_summary.signal_distribution.NEUTRAL,
                            currentData.market_summary.signal_distribution.SELL,
                            currentData.market_summary.signal_distribution.STRONG_SELL
                        ],
                        backgroundColor: [
                            '#10b981', '#34d399', '#6b7280', '#f87171', '#ef4444'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { color: '#9ca3af' } } }
                }
            });

            // Set default stock for charts
            if (currentData.top_picks && currentData.top_picks[0]) {
                updateStockDetails(currentData.top_picks[0]);
            }
        }

        // Run new scan
        function runScan() {
            fetch('/run_scan')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    }
                });
        }

        // Run single stock scan
        function runSingleScan(ticker) {
            fetch(`/scan_stock/${ticker}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateLiveUpdate(`Rescanned ${ticker}: ${data.stock.signal} (Score: ${data.stock.overall_score})`);
                        updateStockDetails(data.stock);
                    }
                });
        }

        // Show stock details
        function showDetails(ticker) {
            const stock = currentData.all_results.find(s => s.ticker === ticker);
            if (stock) {
                updateStockDetails(stock);
                // Scroll to details
                document.getElementById('detail-ticker').scrollIntoView({ behavior: 'smooth' });
            }
        }

        // Update stock details
        function updateStockDetails(stock) {
            document.getElementById('detail-ticker').textContent = stock.ticker;
            document.getElementById('detail-rsi').textContent = stock.technical.rsi.toFixed(1);
            document.getElementById('detail-macd').textContent = stock.technical.macd.toFixed(3);
            document.getElementById('detail-atr').textContent = stock.technical.atr.toFixed(2);
            document.getElementById('detail-momentum').textContent = stock.technical.momentum_pct.toFixed(1) + '%';
            document.getElementById('detail-sentiment').textContent = stock.sentiment.trend + ' (' + stock.sentiment.score.toFixed(3) + ')';
            document.getElementById('detail-pcr').textContent = stock.options.volume_ratio.toFixed(3);
            document.getElementById('detail-volume').textContent = stock.options.total_volume.toLocaleString();
            document.getElementById('detail-position').textContent = stock.risk.position_size + ' shares';
            document.getElementById('detail-stop').textContent = '$' + stock.risk.stop_loss.toFixed(2);
            document.getElementById('detail-take').textContent = '$' + stock.risk.take_profit.toFixed(2);
            document.getElementById('detail-rr').textContent = stock.risk.risk_reward.toFixed(1) + ':1';

            // Update charts
            updateVolumeProfileChart(stock);
            updateTimeframeChart(stock);
        }

        // Update volume profile chart
        function updateVolumeProfileChart(stock) {
            const volumeData = stock.volume_profile;
            if (!volumeData || !volumeData.high_volume_nodes) return;

            const prices = volumeData.high_volume_nodes.map(node => node.price);
            const volumes = volumeData.high_volume_nodes.map(node => node.volume);

            const trace = {
                x: prices,
                y: volumes,
                type: 'bar',
                name: 'Volume',
                marker: { color: '#8b5cf6' }
            };

            const layout = {
                title: `Volume Profile - ${stock.ticker}`,
                xaxis: { title: 'Price', gridcolor: '#374151', color: '#9ca3af' },
                yaxis: { title: 'Volume', gridcolor: '#374151', color: '#9ca3af' },
                plot_bgcolor: '#1f2937',
                paper_bgcolor: '#1f2937',
                font: { color: '#9ca3af' }
            };

            Plotly.newPlot('volumeProfileChart', [trace], layout);
        }

        // Update timeframe chart
        function updateTimeframeChart(stock) {
            const tfData = stock.timeframe_analysis;
            const timeframes = ['15M', '1H', '1D', '3D'];
            
            const alignmentScores = timeframes.map(tf => 
                tfData[tf] ? tfData[tf].alignment_score : 0
            );

            const trace = {
                x: timeframes,
                y: alignmentScores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Alignment Score',
                line: { color: '#3b82f6', width: 3 },
                marker: { size: 10 }
            };

            const layout = {
                title: `Multi-Timeframe Alignment - ${stock.ticker}`,
                xaxis: { title: 'Timeframe', gridcolor: '#374151', color: '#9ca3af' },
                yaxis: { 
                    title: 'Alignment Score', 
                    range: [-1.1, 1.1],
                    gridcolor: '#374151', 
                    color: '#9ca3af'
                },
                shapes: [
                    { type: 'line', x0: -0.5, x1: 3.5, y0: 0, y1: 0, line: { color: '#6b7280', width: 1 } },
                    { type: 'rect', x0: -0.5, x1: 3.5, y0: 0.5, y1: 1.1, fillcolor: 'rgba(16, 185, 129, 0.1)', line: { width: 0 } },
                    { type: 'rect', x0: -0.5, x1: 3.5, y0: -0.5, y1: -1.1, fillcolor: 'rgba(239, 68, 68, 0.1)', line: { width: 0 } }
                ],
                plot_bgcolor: '#1f2937',
                paper_bgcolor: '#1f2937',
                font: { color: '#9ca3af' }
            };

            Plotly.newPlot('timeframeChart', [trace], layout);
        }

        // Update live updates
        function updateLiveUpdate(message) {
            const updatesDiv = document.getElementById('live-updates');
            const now = new Date().toLocaleTimeString();
            const updateItem = `<div class="mb-2 p-2 bg-gray-750 rounded"><span class="text-gray-400 text-sm">[${now}]</span> ${message}</div>`;
            updatesDiv.insertAdjacentHTML('afterbegin', updateItem);
            
            // Limit to 10 updates
            const children = updatesDiv.children;
            if (children.length > 10) {
                updatesDiv.removeChild(children[children.length - 1]);
            }
        }

        // Auto-refresh every 5 minutes
        setInterval(() => {
            updateLiveUpdate('Auto-refresh in progress...');
            runScan();
        }, 300000);

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateLiveUpdate('Dashboard initialized successfully.');
        });
    </script>
</body>
</html>
'''

def run_background_scan():
    """Run scanner in background"""
    global scanner, last_scan_results, last_scan_time
    
    print("Running background market scan...")
    scanner = MarketScanner()
    scan_results = scanner.scan_watchlist()
    last_scan_results = scan_results
    last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Scan complete at {last_scan_time}")
    return scan_results

@app.route('/')
def dashboard():
    """Main dashboard page"""
    global last_scan_results, last_scan_time
    
    if last_scan_results is None:
        # Run initial scan
        scan_thread = threading.Thread(target=run_background_scan)
        scan_thread.start()
        scan_thread.join(timeout=30)
    
    if last_scan_results:
        market_summary = last_scan_results['market_summary']
        top_stocks = last_scan_results['top_picks'][:5] if last_scan_results['top_picks'] else []
        top_stock = top_stocks[0] if top_stocks else None
        
        # Prepare data for template
        template_data = {
            'last_scan_time': last_scan_time or 'Never',
            'market_strength': market_summary.get('market_strength', 0),
            'market_trend': market_summary.get('market_trend', 'NEUTRAL'),
            'stocks_analyzed': last_scan_results.get('stocks_analyzed', 0),
            'bullish_signals': (market_summary.get('signal_distribution', {}).get('STRONG_BUY', 0) + 
                              market_summary.get('signal_distribution', {}).get('BUY', 0)),
            'top_stock': top_stock,
            'top_stocks': top_stocks,
            'selected_ticker': top_stock['ticker'] if top_stock else 'N/A',
            'scan_data': last_scan_results
        }
        
        return render_template_string(HTML_TEMPLATE, **template_data)
    else:
        # Show loading screen
        loading_html = '''
        <!DOCTYPE html>
        <html>
        <head><title>Loading Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-900 text-white h-screen flex items-center justify-center">
            <div class="text-center">
                <div class="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-yellow-400 mx-auto mb-8"></div>
                <h1 class="text-3xl font-bold mb-4">KING DOM TRADING DASHBOARD</h1>
                <p class="text-xl text-gray-400">Running initial market scan...</p>
                <p class="text-gray-500 mt-4">This may take 30-60 seconds</p>
                <script>
                    setTimeout(() => { location.reload(); }, 5000);
                </script>
            </div>
        </body>
        </html>
        '''
        return loading_html

@app.route('/run_scan')
def run_scan():
    """Run a new scan"""
    global scanner, last_scan_results, last_scan_time
    
    def scan_task():
        run_background_scan()
    
    # Run in background thread
    thread = threading.Thread(target=scan_task)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Scan started in background',
        'time': datetime.now().isoformat()
    })

@app.route('/scan_stock/<ticker>')
def scan_single_stock(ticker):
    """Scan a single stock"""
    global scanner
    
    if scanner is None:
        scanner = MarketScanner()
    
    result = scanner.scan_single_stock(ticker)
    
    return jsonify({
        'success': result is not None,
        'stock': result,
        'ticker': ticker
    })

@app.route('/api/scan_data')
def get_scan_data():
    """Get raw scan data as JSON"""
    global last_scan_results
    
    if last_scan_results:
        return jsonify(last_scan_results)
    else:
        return jsonify({'error': 'No scan data available'})

@app.route('/api/stock/<ticker>')
def get_stock_data(ticker):
    """Get data for a specific stock"""
    global last_scan_results
    
    if last_scan_results and 'all_results' in last_scan_results:
        for stock in last_scan_results['all_results']:
            if stock['ticker'] == ticker:
                return jsonify(stock)
    
    return jsonify({'error': 'Stock not found'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'last_scan': last_scan_time,
        'stocks_analyzed': len(last_scan_results.get('all_results', [])) if last_scan_results else 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("KING DOM TRADING - PRO DASHBOARD")
    print("=" * 60)
    print("Dashboard starting on http://localhost:5000")
    print("Initial market scan will run automatically...")
    print("\nFeatures:")
    print("• Real-time market scanning")
    print("• Interactive charts (Volume Profile, Timeframe Alignment)")
    print("• Top 5 trading opportunities")
    print("• Detailed stock analysis")
    print("• Professional risk management")
    print("=" * 60)
    
    # Run initial scan in background
    scan_thread = threading.Thread(target=run_background_scan)
    scan_thread.start()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

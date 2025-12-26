# TradeHandler AI - Trading Agent API

A sophisticated AI-powered trading agent platform built with FastAPI, LangGraph, and Zerodha Kite Connect. The system provides autonomous market scanning, intelligent trade signal generation, risk management, and hybrid human-AI approval workflows.

## 🚀 Features

### Core Capabilities
- **AI Trading Agent**: LangGraph-based intelligent agent that understands natural language queries and executes trading operations
- **Institutional VWAP Strategy**: Primary trading strategy focusing on VWAP-based signals with RSI and candlestick pattern confirmation
- **Autonomous Market Scanning**: Background task that continuously scans the market for trading opportunities
- **Hybrid Approval System**: Human-in-the-loop approval workflow with auto-approval thresholds
- **Real-time WebSocket Updates**: Live updates for approvals, logs, and market data
- **Historical Simulation**: Backtest strategies on historical data with full trade simulation
- **SQLite Database**: Persistent storage for approvals, logs, configurations, and trade history

### Trading Features
- **Multi-Instrument Analysis**: Scan multiple stocks or groups (e.g., "top 10 nifty50 stocks")
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits, circuit breakers
- **Order Management**: Place, modify, cancel orders with automatic SL/TP placement
- **Auto-Cancellation**: Automatically cancels remaining orders when SL or Target executes
- **Prime Session Trading**: Configurable trading windows (default: 10:15 AM - 2:45 PM IST)
- **Intraday Square-off**: Mandatory position closure at 3:15 PM IST

### Technical Indicators
- VWAP (Volume Weighted Average Price)
- RSI (Relative Strength Index)
- EMA (Exponential Moving Average)
- Candlestick Patterns (Rejection, Engulfing, Hammer)

## 📁 Project Structure

```
tradehandler-api/
├── agent/                    # AI Agent core
│   ├── graph.py             # LangGraph workflow definition
│   ├── nodes.py             # Agent nodes (intent, tool selection, execution)
│   ├── config.py            # Centralized configuration
│   ├── approval.py          # Approval queue management
│   ├── autonomous.py        # Autonomous market scanner
│   ├── safety.py            # Risk and safety checks
│   ├── ws_manager.py        # WebSocket management
│   └── tools/               # Trading tools
│       ├── trading_opportunities_tool.py  # Main VWAP strategy
│       ├── kite_tools.py    # Zerodha Kite Connect integration
│       ├── market_tools.py  # Market data tools
│       ├── risk_tools.py    # Risk calculation tools
│       ├── simulation_tools.py  # Historical simulation
│       └── ...
├── database/                # SQLite database layer
│   ├── models.py           # Pydantic models
│   ├── connection.py      # Database connection (thread-safe)
│   └── repositories.py    # Data access layer
├── utils/                   # Utilities
│   ├── kite_utils.py      # Kite Connect helpers
│   └── logger.py          # File-based logging
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
└── .env                    # Environment configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Zerodha Kite Connect API credentials
- OpenAI/Anthropic API key (or Ollama for local LLM)

### Setup

1. **Clone and navigate to the project**
   ```bash
   cd tradehandler-api
   ```

2. **Create virtual environment**
   ```bash
   python3.12 -m venv algo-env
   source algo-env/bin/activate  # On Windows: algo-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the `tradehandler-api` directory:
   ```env
   # Kite Connect
   KITE_API_KEY=your_api_key
   KITE_API_SECRET=your_api_secret
   KITE_ACCESS_TOKEN=your_access_token
   
   # LLM Configuration
   LLM_PROVIDER=openai  # or anthropic, ollama
   OPENAI_API_KEY=your_openai_key
   AGENT_MODEL=gpt-4-turbo-preview
   AGENT_TEMPERATURE=0.3
   
   # Trading Configuration
   TRADING_CAPITAL=200000.0
   RISK_PER_TRADE_PCT=1.0
   REWARD_PER_TRADE_PCT=3.0
   AUTO_TRADE_THRESHOLD=5000.0
   MAX_POSITION_SIZE=200000.0
   DAILY_LOSS_LIMIT=5000.0
   
   # Strategy Parameters
   VWAP_PROXIMITY_PCT=0.5
   PRIME_SESSION_START=10:15
   PRIME_SESSION_END=14:45
   INTRADAY_SQUARE_OFF_TIME=15:15
   
   # Autonomous Mode
   AUTONOMOUS_MODE=false
   AUTONOMOUS_SCAN_INTERVAL_MINS=5
   AUTONOMOUS_TARGET_GROUP=top 10 nifty50 stocks
   ```

5. **Initialize database**
   The database is automatically initialized on first run. SQLite database is created at `data/tradehandler.db`.

## 🚀 Running the Application

### Development Mode
```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Agent Endpoints
- `POST /agent/chat` - Chat with the AI agent
- `GET /agent/status` - Get agent status
- `GET /agent/config` - Get agent configuration
- `POST /agent/config` - Update agent configuration
- `GET /agent/approvals` - Get pending approvals
- `GET /agent/approved-trades` - Get approved trades
- `POST /agent/approve/{approval_id}` - Approve a trade
- `POST /agent/reject/{approval_id}` - Reject a trade

### WebSocket Endpoints
- `WS /ws/agent` - Real-time agent updates (approvals, logs, status)

### Market Data Endpoints
- `GET /api/quote/{exchange}/{symbol}` - Get current quote
- `GET /api/historical/{exchange}/{symbol}` - Get historical data
- `GET /api/positions` - Get current positions
- `GET /api/orders` - Get order history

## 🤖 AI Agent Usage

### Natural Language Queries

The agent understands natural language and can handle queries like:

**Trading Opportunities**
```
"Find trading opportunities in RELIANCE today"
"What trades would I have taken in top 10 nifty50 stocks yesterday?"
"Show me VWAP signals in TCS for the last week"
```

**Market Analysis**
```
"What's the current price of RELIANCE?"
"Analyze the trend in NIFTY"
"Calculate RSI for HDFCBANK"
```

**Simulation**
```
"Download 1-minute historical data for top 10 nifty50 stocks for yesterday"
"Run a simulation on the downloaded local data"
```

**Portfolio Management**
```
"Show my current positions"
"What's my account balance?"
"Get portfolio summary"
```

## 📊 Institutional VWAP Strategy

The primary trading strategy focuses on:

1. **VWAP Proximity**: Price must be within 0.5% (single) or 0.75% (group) of VWAP
2. **RSI Pullback**: 
   - BUY: RSI < 50 and turning up
   - SELL: RSI > 50 and turning down
3. **Candle Patterns**: Rejection candles (35% shadow) or Engulfing patterns
4. **Prime Session**: Trades only during 10:15 AM - 2:45 PM IST
5. **Risk Management**: 1% risk per trade, 3% reward target (3:1 R:R ratio)

## 🔒 Risk Management

- **Position Sizing**: Dynamic position sizing based on available capital
- **Stop Loss**: Automatic SL placement at VWAP ± 0.2% or entry ± 0.5%
- **Take Profit**: Target set at 3x risk distance
- **Daily Loss Limit**: Trading stops if daily loss exceeds limit
- **Circuit Breaker**: Automatic halt if cumulative loss exceeds threshold
- **Max Position Size**: Limits maximum position value
- **Max Trades Per Day**: Prevents overtrading

## 🗄️ Database Schema

The SQLite database stores:
- **agent_approvals**: Trade approvals and decisions
- **agent_logs**: Activity logs with timestamps
- **agent_config**: Configuration settings
- **simulation_results**: Backtest results
- **tool_executions**: Tool input/output logs
- **chat_messages**: Conversation history

## 📝 Logging

Logs are written to:
- `logs/agent.log` - Agent activity logs
- `logs/tools.log` - Tool execution logs (inputs/outputs)

Logs are automatically rotated (5MB max, 5 backups).

## 🔧 Configuration

All configuration is managed through:
1. `.env` file (persistent storage)
2. `/agent/config` API endpoint (runtime updates)
3. Frontend configuration page (UI-based updates)

Configuration includes:
- LLM provider and model settings
- Trading capital and risk parameters
- Strategy parameters (VWAP proximity, session times)
- Autonomous mode settings
- Safety limits and circuit breakers

## 🧪 Testing

Test scripts available:
- `test_database.py` - Database functionality tests
- `test_threading.py` - Thread-safety tests

## 📚 Documentation

- `AGENT_PROMPT_EXAMPLES.md` - Example queries for the AI agent
- `CHART_REPLAY_IMPLEMENTATION.md` - Chart replay feature documentation

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep Kite Connect credentials secure
- Use environment variables for sensitive data
- API keys should be rotated regularly

## 🐛 Troubleshooting

### Common Issues

1. **"Too many requests" error**
   - Solution: Instrument resolution is now cached. Restart the server to clear cache.

2. **Database threading errors**
   - Solution: Database uses thread-local connections. Ensure you're using the latest version.

3. **Signals from old data**
   - Solution: Live mode now only uses the most recent candle. Check logs for "LIVE MODE" confirmation.

4. **Missing order IDs in approvals**
   - Solution: Old approvals won't have order IDs. New approvals will include them automatically.

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

1. Follow Python PEP 8 style guidelines
2. Add type hints to all functions
3. Write docstrings for all modules and functions
4. Test thoroughly before submitting

## 📞 Support

For issues or questions, check the logs in `logs/` directory and review the error messages in the API responses.

---

**Built with ❤️ using FastAPI, LangGraph, and Zerodha Kite Connect**

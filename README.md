# TradeHandler AI - Trading Agent API

A sophisticated, enterprise-grade AI-powered trading agent platform built with FastAPI, LangGraph, and Zerodha Kite Connect. The system provides autonomous market scanning, intelligent trade signal generation, risk management, and hybrid human-AI approval workflows with comprehensive error handling, logging, and monitoring.

## 🚀 Features

### Enterprise-Level Architecture
- **API Versioning**: RESTful API with versioning support (`/api/v1/`)
- **Request/Response Standardization**: Consistent response format with request ID tracking
- **Comprehensive Error Handling**: Global exception handling with structured error responses
- **Request Tracing**: Unique request ID for every request for end-to-end debugging
- **Structured Logging**: Request/response logging with performance metrics
- **Input Validation**: Pydantic-based request validation for type safety
- **Auto-Generated Documentation**: OpenAPI/Swagger docs at `/docs` and ReDoc at `/redoc`

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
├── api/                      # API routes (enterprise structure)
│   └── v1/                  # API version 1
│       ├── __init__.py     # API router aggregation
│       ├── health.py       # Health check endpoints
│       └── routes/          # Modular route handlers
│           └── agent.py    # Agent endpoints router
├── core/                     # Core enterprise components
│   ├── config.py           # Application settings
│   ├── dependencies.py     # Dependency injection
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── responses.py        # Standardized response models
│   └── validators.py      # Input validation utilities
├── middleware/              # Enterprise middleware stack
│   ├── error_handler.py   # Global error handling
│   ├── logging.py         # Request/response logging
│   ├── request_id.py      # Request ID tracking
│   └── __init__.py        # Middleware exports
├── schemas/                 # Request/response schemas
│   └── agent.py           # Agent API schemas
├── database/                # SQLite database layer
│   ├── models.py           # Pydantic models
│   ├── connection.py       # Database connection (thread-safe)
│   └── repositories.py     # Data access layer
├── utils/                   # Utilities
│   ├── kite_utils.py       # Kite Connect helpers
│   └── logger.py          # File-based logging
├── main.py                  # FastAPI application (enterprise setup)
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── ENTERPRISE_UPGRADE.md    # Enterprise upgrade documentation
└── CHANGELOG_ENTERPRISE.md  # Enterprise upgrade changelog
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

### API Documentation
Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs` - Interactive API documentation
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation format

## 📡 API Endpoints

### Enterprise API Endpoints (Recommended)
All endpoints under `/api/v1/` with standardized request/response format:

**Agent Endpoints**
- `POST /api/v1/agent/chat` - Chat with the AI agent
- `GET /api/v1/agent/status` - Get agent status
- `GET /api/v1/agent/config` - Get agent configuration
- `POST /api/v1/agent/config` - Update agent configuration
- `GET /api/v1/agent/approvals` - Get pending approvals
- `GET /api/v1/agent/approved-trades` - Get approved trades
- `POST /api/v1/agent/approve/{approval_id}` - Approve a trade
- `POST /api/v1/agent/reject/{approval_id}` - Reject a trade

**Health & Monitoring**
- `GET /api/v1/health/health` - Health check endpoint
- `GET /api/v1/health/ready` - Readiness check (Kubernetes)
- `GET /api/v1/health/live` - Liveness check (Kubernetes)

### Legacy Endpoints (Backward Compatible)
All legacy endpoints at `/agent/*` continue to work for backward compatibility.

### WebSocket Endpoints
- `WS /ws/agent` - Real-time agent updates (approvals, logs, status)

### API Documentation
- `GET /docs` - Interactive Swagger/OpenAPI documentation
- `GET /redoc` - Alternative ReDoc documentation
- `GET /openapi.json` - OpenAPI schema JSON

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

## 📝 Logging & Monitoring

### File-Based Logging
Logs are written to:
- `logs/agent.log` - Agent activity logs
- `logs/tools.log` - Tool execution logs (inputs/outputs)

Logs are automatically rotated (5MB max, 5 backups).

### Request/Response Logging
- All HTTP requests and responses are logged with:
  - Request ID for tracing
  - HTTP method and path
  - Response status code
  - Processing time (in `X-Process-Time` header)
- Request IDs are included in all responses and error messages

### Error Logging
- Structured error logging with error codes
- Request ID included in all error logs
- Debug mode provides detailed stack traces

## 🔧 Configuration

All configuration is managed through:
1. `.env` file (persistent storage)
2. `/api/v1/agent/config` API endpoint (runtime updates)
3. Frontend configuration page (UI-based updates)

### Configuration Categories

**Application Settings** (via `core/config.py`):
- `ENVIRONMENT` - development/staging/production
- `DEBUG` - Enable debug mode
- `LOG_LEVEL` - Logging verbosity
- `CORS_ORIGINS` - Allowed CORS origins
- `SECRET_KEY` - Application secret key

**Trading Configuration**:
- LLM provider and model settings
- Trading capital and risk parameters
- Strategy parameters (VWAP proximity, session times)
- Autonomous mode settings
- Safety limits and circuit breakers
- GTT order settings

## 🧪 Testing

Test scripts available:
- `test_database.py` - Database functionality tests
- `test_threading.py` - Thread-safety tests

### Testing API Endpoints

Use the interactive Swagger documentation at `/docs` to test endpoints directly, or use curl:

```bash
# Test chat endpoint
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show my positions", "session_id": "test"}'

# Test health check
curl "http://localhost:8000/api/v1/health/health"
```

## 📚 Documentation

### Enterprise Documentation
- `ENTERPRISE_UPGRADE.md` - Comprehensive enterprise-level upgrade documentation
- `CHANGELOG_ENTERPRISE.md` - Enterprise upgrade changelog

### Feature Documentation
- `AGENT_PROMPT_EXAMPLES.md` - Example queries for the AI agent
- `CHART_REPLAY_IMPLEMENTATION.md` - Chart replay feature documentation
- `GTT_IMPLEMENTATION_SUMMARY.md` - GTT (Good Till Triggered) orders documentation

### API Documentation
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep Kite Connect credentials secure
- Use environment variables for sensitive data
- API keys should be rotated regularly
- Request ID tracking helps with security auditing
- All errors are logged with request context for forensics

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

5. **Request ID not appearing in responses**
   - Solution: Ensure `RequestIDMiddleware` is properly configured. Check middleware order in `main.py`.

6. **API endpoint not found**
   - Solution: Use `/api/v1/agent/*` for new endpoints. Legacy `/agent/*` endpoints still work.

### Debugging Tips

- **Check Request IDs**: Every request has a unique ID. Use it to trace requests across logs.
- **Review Logs**: Check `logs/agent.log` for detailed activity logs with request IDs.
- **API Documentation**: Use `/docs` to explore available endpoints and test them.
- **Error Responses**: All errors include `request_id` and `error_code` for easier debugging.

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

1. Follow Python PEP 8 style guidelines
2. Add type hints to all functions
3. Write docstrings for all modules and functions
4. Test thoroughly before submitting
5. Use the enterprise API structure (`/api/v1/`) for new endpoints
6. Follow the standardized response format (`APIResponse`, `SuccessResponse`, `ErrorResponse`)
7. Include request ID tracking in all new endpoints
8. Add proper error handling using custom exceptions from `core/exceptions.py`

## 📞 Support

For issues or questions, check the logs in `logs/` directory and review the error messages in the API responses.

## 🏗️ Enterprise Architecture

### Middleware Stack
The application uses a comprehensive middleware stack (applied in order):
1. **RequestIDMiddleware** - Adds unique request ID to every request
2. **LoggingMiddleware** - Logs all HTTP requests/responses with timing
3. **ErrorHandlerMiddleware** - Global exception handling with structured responses
4. **CORSMiddleware** - Cross-origin resource sharing

### Response Format
All API responses follow a standardized format:

**Success Response:**
```json
{
  "status": "success",
  "message": "Optional message",
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00",
  "request_id": "uuid-here"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Error message",
  "error_code": "ERROR_CODE",
  "details": { ... },
  "timestamp": "2024-01-01T00:00:00",
  "request_id": "uuid-here"
}
```

### Error Codes
- `VALIDATION_ERROR` - Input validation failed
- `AUTHENTICATION_ERROR` - Authentication required
- `NOT_FOUND` - Resource not found
- `BUSINESS_LOGIC_ERROR` - Business rule violation
- `EXTERNAL_API_ERROR` - External API (Kite) error
- `INTERNAL_SERVER_ERROR` - Unexpected server error

### Request ID Tracking
- Every request automatically gets a unique `X-Request-ID` header
- Request ID is included in:
  - Response headers
  - All log entries
  - Error responses
  - WebSocket messages

This enables end-to-end request tracing for debugging and monitoring.

---

**Built with ❤️ using FastAPI, LangGraph, and Zerodha Kite Connect**

**Enterprise-Grade Architecture | Production-Ready | Fully Documented**

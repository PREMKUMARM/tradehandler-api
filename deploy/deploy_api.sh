#!/bin/bash

# =================================================================
# AlgoFeast Backend Deployment Script
# This script connects to EC2, pulls latest code, and restarts the API service.
# Does NOT run pip install by default (full reinstall can OOM small instances).
# To install deps manually when requirements.txt changes:
#   DEPLOY_INSTALL_DEPS=1 bash deploy/deploy_api.sh
# =================================================================

# --- CONFIGURATION ---
# Path to your .pem file (Change this to your actual path)
PEM_FILE="/Users/premkumar/Documents/vibefno.pem"

# EC2 Connection Details
EC2_USER="ec2-user"
EC2_IP="ec2-3-108-61-102.ap-south-1.compute.amazonaws.com"

# Remote Paths
REMOTE_API_PATH="/home/ec2-user/algofeast-workspace/algofeast-api"
SERVICE_NAME="algofeast-api"
LOCAL_ENV_FILE=".env"
REMOTE_ENV_FILE="$REMOTE_API_PATH/.env"

# Binance-related environment variables to sync
BINANCE_VARS=("BINANCE_API_KEY" "BINANCE_API_SECRET" "BINANCE_SYMBOLS")

# --- EXECUTION ---

# Exit on any error
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "📤 Syncing local workspace to git..."
cd "$REPO_ROOT"

git add .

if git diff --cached --quiet; then
    echo "  ℹ️  No changes to commit."
else
    COMMIT_MSG="${DEPLOY_COMMIT_MESSAGE:-Deploy API $(date +%Y-%m-%d\ %H:%M:%S)}"
    git commit -m "$COMMIT_MSG"
    echo "  ✓ Committed: $COMMIT_MSG"
fi

git push
echo "  ✓ Pushed to remote."

echo ""
echo "🚀 Starting backend deployment to $EC2_IP..."

# Extract Binance + Telegram + FCM environment variables from local .env
echo "🔐 Extracting Binance, Telegram, and FCM variables from local .env..."
BINANCE_API_KEY_VAL=""
BINANCE_API_SECRET_VAL=""
BINANCE_SYMBOLS_VAL=""
TELEGRAM_BOT_TOKEN_VAL=""
TELEGRAM_CHAT_ID_VAL=""
FCM_PROJECT_ID_VAL=""
FCM_SERVICE_ACCOUNT_JSON_VAL=""

if [ -f "$LOCAL_ENV_FILE" ]; then
    BINANCE_API_KEY_VAL=$(grep "^BINANCE_API_KEY=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    BINANCE_API_SECRET_VAL=$(grep "^BINANCE_API_SECRET=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    BINANCE_SYMBOLS_VAL=$(grep "^BINANCE_SYMBOLS=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    TELEGRAM_BOT_TOKEN_VAL=$(grep "^TELEGRAM_BOT_TOKEN=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    TELEGRAM_CHAT_ID_VAL=$(grep "^TELEGRAM_CHAT_ID=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    FCM_PROJECT_ID_VAL=$(grep "^FCM_PROJECT_ID=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    FCM_SERVICE_ACCOUNT_JSON_VAL=$(grep "^FCM_SERVICE_ACCOUNT_JSON=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    
    [ -n "$BINANCE_API_KEY_VAL" ] && echo "  ✓ Found BINANCE_API_KEY"
    [ -n "$BINANCE_API_SECRET_VAL" ] && echo "  ✓ Found BINANCE_API_SECRET"
    [ -n "$BINANCE_SYMBOLS_VAL" ] && echo "  ✓ Found BINANCE_SYMBOLS"
    [ -n "$TELEGRAM_BOT_TOKEN_VAL" ] && echo "  ✓ Found TELEGRAM_BOT_TOKEN"
    [ -n "$TELEGRAM_CHAT_ID_VAL" ] && echo "  ✓ Found TELEGRAM_CHAT_ID"
    [ -n "$FCM_PROJECT_ID_VAL" ] && echo "  ✓ Found FCM_PROJECT_ID"
    [ -n "$FCM_SERVICE_ACCOUNT_JSON_VAL" ] && echo "  ✓ Found FCM_SERVICE_ACCOUNT_JSON"
else
    echo "  ⚠️  Local .env file not found, skipping environment variable sync"
fi

_read_local_env() {
    local key="$1"
    if [ -f "$REPO_ROOT/$LOCAL_ENV_FILE" ]; then
        grep "^${key}=" "$REPO_ROOT/$LOCAL_ENV_FILE" 2>/dev/null | tail -1 | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    fi
}

ENTRY_INITIAL_RR_VAL="$(_read_local_env ENTRY_INITIAL_RR)"
ENTRY_VALIDATION_SKIP_REWARD_VAL="$(_read_local_env ENTRY_VALIDATION_SKIP_REWARD)"
STEPPED_RR_TRAIL_VAL="$(_read_local_env STEPPED_RR_TRAIL)"
MOMENTUM_TRAIL_BREAKEVEN_PCT_VAL="$(_read_local_env MOMENTUM_TRAIL_BREAKEVEN_PCT)"
MOMENTUM_TRAIL_BREAKEVEN_R_FRACTION_VAL="$(_read_local_env MOMENTUM_TRAIL_BREAKEVEN_R_FRACTION)"
MOMENTUM_TRAIL_BREAKEVEN_BUFFER_VAL="$(_read_local_env MOMENTUM_TRAIL_BREAKEVEN_BUFFER)"
TRAIL_PARTIAL_EXIT_ENABLED_VAL="$(_read_local_env TRAIL_PARTIAL_EXIT_ENABLED)"
TRAIL_PARTIAL_EXIT_PCT_VAL="$(_read_local_env TRAIL_PARTIAL_EXIT_PCT)"
TRAIL_ACTIVATION_HOLD_SEC_VAL="$(_read_local_env TRAIL_ACTIVATION_HOLD_SEC)"
TRAIL_REQUIRE_5M_CLOSE_VAL="$(_read_local_env TRAIL_REQUIRE_5M_CLOSE)"
TRAIL_TIME_STOP_ENABLED_VAL="$(_read_local_env TRAIL_TIME_STOP_ENABLED)"
TRAIL_TIME_STOP_MINUTES_VAL="$(_read_local_env TRAIL_TIME_STOP_MINUTES)"
TRAIL_TIME_STOP_BEFORE_IST_VAL="$(_read_local_env TRAIL_TIME_STOP_BEFORE_IST)"
TRAIL_TIME_STOP_ONLY_WITHOUT_1R_VAL="$(_read_local_env TRAIL_TIME_STOP_ONLY_WITHOUT_1R)"
TRAIL_STALE_ALERT_MIN_VAL="$(_read_local_env TRAIL_STALE_ALERT_MIN)"
TRAIL_GTT_FAIL_ALERT_THRESHOLD_VAL="$(_read_local_env TRAIL_GTT_FAIL_ALERT_THRESHOLD)"
MOMENTUM_TRAIL_POLL_SEC_VAL="$(_read_local_env MOMENTUM_TRAIL_POLL_SEC)"
MOMENTUM_TRAIL_ENABLED_VAL="$(_read_local_env MOMENTUM_TRAIL_ENABLED)"
COMMODITY_AUTO_MIN_ENTRY_SCORE_VAL="$(_read_local_env COMMODITY_AUTO_MIN_ENTRY_SCORE)"
DHAN_ACCESS_TOKEN_VAL="$(_read_local_env DHAN_ACCESS_TOKEN)"
DHAN_CLIENT_ID_VAL="$(_read_local_env DHAN_CLIENT_ID)"
NIFTY_BOUNCE_RECOVERY_PTS_VAL="$(_read_local_env NIFTY_BOUNCE_RECOVERY_PTS)"
NIFTY_FADE_RECOVERY_PTS_VAL="$(_read_local_env NIFTY_FADE_RECOVERY_PTS)"
NIFTY_BB_TOUCH_BUFFER_PTS_VAL="$(_read_local_env NIFTY_BB_TOUCH_BUFFER_PTS)"
NIFTY_CHOP_RANGE_PTS_VAL="$(_read_local_env NIFTY_CHOP_RANGE_PTS)"
NIFTY_CHOP_MID_LOW_VAL="$(_read_local_env NIFTY_CHOP_MID_LOW)"
NIFTY_CHOP_MID_HIGH_VAL="$(_read_local_env NIFTY_CHOP_MID_HIGH)"
NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC_VAL="$(_read_local_env NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC)"
V2_AUTO_MAX_LOTS_VAL="$(_read_local_env V2_AUTO_MAX_LOTS)"
V2_AUTO_REQUIRE_PREFERRED_BB_VAL="$(_read_local_env V2_AUTO_REQUIRE_PREFERRED_BB)"
V2_WATCH_REENTRY_COOLDOWN_SEC_VAL="$(_read_local_env V2_WATCH_REENTRY_COOLDOWN_SEC)"
V2_WATCH_MAX_DIRECTION_FLIPS_VAL="$(_read_local_env V2_WATCH_MAX_DIRECTION_FLIPS)"
INVALIDATION_SPOT_TRIGGER_BUFFER_VAL="$(_read_local_env INVALIDATION_SPOT_TRIGGER_BUFFER)"

# 20rupees strategy / Dhan backtest (sync all non-empty keys from local .env)
STRATEGY_ENV_KEYS=(
    EXIT_MODEL
    NIFTY_DUCKDB_PATH
    ENTRY_REQUIRE_MOMENTUM_BAR
    ENTRY_REQUIRE_INDEX_MOMENTUM
    ENTRY_REQUIRE_DAY_ALIGNED
    ENTRY_REQUIRE_CONFIRMATION_CANDLE
    ENTRY_MID_BAND_ONLY
    ENTRY_MAX_CLOSE_POSITION_IN_BAR
    ENTRY_BAND_LOW
    ENTRY_BAND_HIGH
    SENSEX_ENTRY_START_HOUR
    SENSEX_ENTRY_START_MINUTE
    SENSEX_ENTRY_CUTOFF_HOUR
    SENSEX_ENTRY_CUTOFF_MINUTE
    MAX_TRADES_PER_CONTRACT_PER_DAY
    MAX_TRADES_PER_SESSION_DAY
    ENTRY_BLOCK_REENTRY_AFTER_LOSS
    ENTRY_BLOCK_REENTRY_AFTER_BREAKEVEN
    EXIT_T1_CLOSE_CONFIRM
    ENTRY_SCAN_WARMUP_MIN
    ENTRY_MIN_DAY_MOVE_PTS
    ENTRY_CHASE_DAY_PTS
    ENTRY_CHASE_MAX_BAR_MOVE_PTS
    ENTRY_CHASE_MIN_BAR_MOVE_PTS
    ENTRY_PE_BOUNCE_RECOVERY_MAX_PCT
    SENSEX_ENTRY_MIN_DAY_MOVE_PTS
    SENSEX_ENTRY_CHASE_DAY_PTS
    SENSEX_ENTRY_CHASE_MAX_BAR_MOVE_PTS
    SENSEX_ENTRY_CHASE_MIN_BAR_MOVE_PTS
    SENSEX_ENTRY_PE_BOUNCE_RECOVERY_MAX_PCT
)

STRATEGY_ENV_UPSERT_LINES=""
for _key in "${STRATEGY_ENV_KEYS[@]}"; do
    _val="$(_read_local_env "$_key")"
    if [ -n "$_val" ]; then
        # Escape single quotes for remote bash heredoc
        _val_esc="${_val//\'/\'\\\'\'}"
        STRATEGY_ENV_UPSERT_LINES="${STRATEGY_ENV_UPSERT_LINES}
    upsert_env_line \"${_key}\" '${_val_esc}'"
    fi
done

# Normalize FCM JSON path for EC2:
# - If your local .env uses an absolute path, we still deploy the file into the API directory
#   and write FCM_SERVICE_ACCOUNT_JSON as a repo-relative path (basename) on the server.
FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE="$FCM_SERVICE_ACCOUNT_JSON_VAL"
if [[ "$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE" = /* ]]; then
    FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE="$(basename "$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE")"
fi

# If FCM JSON is a relative path, resolve it from repo root (this script lives in deploy/)
FCM_JSON_LOCAL_PATH=""
if [ -n "$FCM_SERVICE_ACCOUNT_JSON_VAL" ]; then
    if [[ "$FCM_SERVICE_ACCOUNT_JSON_VAL" = /* ]]; then
        FCM_JSON_LOCAL_PATH="$FCM_SERVICE_ACCOUNT_JSON_VAL"
    else
        FCM_JSON_LOCAL_PATH="$REPO_ROOT/$FCM_SERVICE_ACCOUNT_JSON_VAL"
    fi
fi

# If you have generated a newer key file locally (different filename), you can point the deploy
# at it and still keep the remote filename stable via FCM_SERVICE_ACCOUNT_JSON in .env.
# This copies the CONTENTS of the new key to the server path expected by the API.
FCM_JSON_LOCAL_SOURCE_PATH="$FCM_JSON_LOCAL_PATH"
FCM_JSON_LOCAL_PREFERRED="$REPO_ROOT/algofeast-notify-firebase-adminsdk-fbsvc-1d9ac17cd0.json"
if [ -f "$FCM_JSON_LOCAL_PREFERRED" ]; then
    FCM_JSON_LOCAL_SOURCE_PATH="$FCM_JSON_LOCAL_PREFERRED"
fi

if [ -n "$FCM_SERVICE_ACCOUNT_JSON_VAL" ]; then
    if [ ! -f "$FCM_JSON_LOCAL_PATH" ]; then
        echo "  ⚠️  FCM service account JSON not found at: $FCM_JSON_LOCAL_PATH"
        echo "      (Will still sync FCM_PROJECT_ID if present, but push sending may be disabled until the JSON exists on EC2.)"
    else
        echo "  ✓ Resolved FCM JSON local path: $FCM_JSON_LOCAL_PATH"
    fi
fi

# Connect to EC2 and sync environment variables first
echo "🔐 Syncing Binance, Telegram, and FCM environment variables to EC2..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" bash << EOF
    set -e
    
    REMOTE_ENV_FILE="$REMOTE_ENV_FILE"
    BINANCE_API_KEY_VAL='$BINANCE_API_KEY_VAL'
    BINANCE_API_SECRET_VAL='$BINANCE_API_SECRET_VAL'
    BINANCE_SYMBOLS_VAL='$BINANCE_SYMBOLS_VAL'
    TELEGRAM_BOT_TOKEN_VAL='$TELEGRAM_BOT_TOKEN_VAL'
    TELEGRAM_CHAT_ID_VAL='$TELEGRAM_CHAT_ID_VAL'
    FCM_PROJECT_ID_VAL='$FCM_PROJECT_ID_VAL'
    FCM_SERVICE_ACCOUNT_JSON_VAL='$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE'
    
    # Replace or append KEY=value (values must not contain single quotes — typical API tokens are fine)
    upsert_env_line() {
        local key="\$1"
        local val="\$2"
        local f="\$REMOTE_ENV_FILE"
        [ -z "\$val" ] && return 0
        touch "\$f"
        if grep -q "^\${key}=" "\$f" 2>/dev/null; then
            grep -v "^\${key}=" "\$f" > "\${f}.new" || true
            mv "\${f}.new" "\$f"
        fi
        echo "\${key}=\${val}" >> "\$f"
        echo "  ✓ Synced \${key}"
    }
    
    echo "🔐 Syncing environment variables to remote .env..."
    if [ ! -f "\$REMOTE_ENV_FILE" ]; then
        echo "  📝 Creating remote .env file..."
        touch "\$REMOTE_ENV_FILE"
    fi
    
    # Sync BINANCE_API_KEY
    if [ -n "\$BINANCE_API_KEY_VAL" ]; then
        if grep -q "^BINANCE_API_KEY=" "\$REMOTE_ENV_FILE" 2>/dev/null; then
            echo "  ✓ BINANCE_API_KEY already exists in remote .env (skipping)"
        else
            echo "  ➕ Adding BINANCE_API_KEY to remote .env..."
            echo "BINANCE_API_KEY=\$BINANCE_API_KEY_VAL" >> "\$REMOTE_ENV_FILE"
        fi
    fi
    
    # Sync BINANCE_API_SECRET
    if [ -n "\$BINANCE_API_SECRET_VAL" ]; then
        if grep -q "^BINANCE_API_SECRET=" "\$REMOTE_ENV_FILE" 2>/dev/null; then
            echo "  ✓ BINANCE_API_SECRET already exists in remote .env (skipping)"
        else
            echo "  ➕ Adding BINANCE_API_SECRET to remote .env..."
            echo "BINANCE_API_SECRET=\$BINANCE_API_SECRET_VAL" >> "\$REMOTE_ENV_FILE"
        fi
    fi
    
    # Sync BINANCE_SYMBOLS
    if [ -n "\$BINANCE_SYMBOLS_VAL" ]; then
        if grep -q "^BINANCE_SYMBOLS=" "\$REMOTE_ENV_FILE" 2>/dev/null; then
            echo "  ✓ BINANCE_SYMBOLS already exists in remote .env (skipping)"
        else
            echo "  ➕ Adding BINANCE_SYMBOLS to remote .env..."
            echo "BINANCE_SYMBOLS=\$BINANCE_SYMBOLS_VAL" >> "\$REMOTE_ENV_FILE"
        fi
    fi
    
    # Telegram — always upsert when local value is non-empty (keeps EC2 in sync with your laptop .env)
    upsert_env_line "TELEGRAM_BOT_TOKEN" "\$TELEGRAM_BOT_TOKEN_VAL"
    upsert_env_line "TELEGRAM_CHAT_ID" "\$TELEGRAM_CHAT_ID_VAL"

    # FCM — upsert when local value is non-empty
    upsert_env_line "FCM_PROJECT_ID" "\$FCM_PROJECT_ID_VAL"
    upsert_env_line "FCM_SERVICE_ACCOUNT_JSON" "\$FCM_SERVICE_ACCOUNT_JSON_VAL"

    # Trail / 1:1 entry policy (from local .env)
    upsert_env_line "ENTRY_INITIAL_RR" '$ENTRY_INITIAL_RR_VAL'
    upsert_env_line "ENTRY_VALIDATION_SKIP_REWARD" '$ENTRY_VALIDATION_SKIP_REWARD_VAL'
    upsert_env_line "STEPPED_RR_TRAIL" '$STEPPED_RR_TRAIL_VAL'
    upsert_env_line "MOMENTUM_TRAIL_BREAKEVEN_PCT" '$MOMENTUM_TRAIL_BREAKEVEN_PCT_VAL'
    upsert_env_line "MOMENTUM_TRAIL_BREAKEVEN_R_FRACTION" '$MOMENTUM_TRAIL_BREAKEVEN_R_FRACTION_VAL'
    upsert_env_line "MOMENTUM_TRAIL_BREAKEVEN_BUFFER" '$MOMENTUM_TRAIL_BREAKEVEN_BUFFER_VAL'
    upsert_env_line "TRAIL_PARTIAL_EXIT_ENABLED" '$TRAIL_PARTIAL_EXIT_ENABLED_VAL'
    upsert_env_line "TRAIL_PARTIAL_EXIT_PCT" '$TRAIL_PARTIAL_EXIT_PCT_VAL'
    upsert_env_line "TRAIL_ACTIVATION_HOLD_SEC" '$TRAIL_ACTIVATION_HOLD_SEC_VAL'
    upsert_env_line "TRAIL_REQUIRE_5M_CLOSE" '$TRAIL_REQUIRE_5M_CLOSE_VAL'
    upsert_env_line "TRAIL_TIME_STOP_ENABLED" '$TRAIL_TIME_STOP_ENABLED_VAL'
    upsert_env_line "TRAIL_TIME_STOP_MINUTES" '$TRAIL_TIME_STOP_MINUTES_VAL'
    upsert_env_line "TRAIL_TIME_STOP_BEFORE_IST" '$TRAIL_TIME_STOP_BEFORE_IST_VAL'
    upsert_env_line "TRAIL_TIME_STOP_ONLY_WITHOUT_1R" '$TRAIL_TIME_STOP_ONLY_WITHOUT_1R_VAL'
    upsert_env_line "TRAIL_STALE_ALERT_MIN" '$TRAIL_STALE_ALERT_MIN_VAL'
    upsert_env_line "TRAIL_GTT_FAIL_ALERT_THRESHOLD" '$TRAIL_GTT_FAIL_ALERT_THRESHOLD_VAL'
    upsert_env_line "MOMENTUM_TRAIL_POLL_SEC" '$MOMENTUM_TRAIL_POLL_SEC_VAL'
    upsert_env_line "MOMENTUM_TRAIL_ENABLED" '$MOMENTUM_TRAIL_ENABLED_VAL'
    upsert_env_line "COMMODITY_AUTO_MIN_ENTRY_SCORE" '$COMMODITY_AUTO_MIN_ENTRY_SCORE_VAL'
    upsert_env_line "DHAN_ACCESS_TOKEN" '$DHAN_ACCESS_TOKEN_VAL'
    upsert_env_line "DHAN_CLIENT_ID" '$DHAN_CLIENT_ID_VAL'
    upsert_env_line "NIFTY_BOUNCE_RECOVERY_PTS" '$NIFTY_BOUNCE_RECOVERY_PTS_VAL'
    upsert_env_line "NIFTY_FADE_RECOVERY_PTS" '$NIFTY_FADE_RECOVERY_PTS_VAL'
    upsert_env_line "NIFTY_BB_TOUCH_BUFFER_PTS" '$NIFTY_BB_TOUCH_BUFFER_PTS_VAL'
    upsert_env_line "NIFTY_CHOP_RANGE_PTS" '$NIFTY_CHOP_RANGE_PTS_VAL'
    upsert_env_line "NIFTY_CHOP_MID_LOW" '$NIFTY_CHOP_MID_LOW_VAL'
    upsert_env_line "NIFTY_CHOP_MID_HIGH" '$NIFTY_CHOP_MID_HIGH_VAL'
    upsert_env_line "NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC" '$NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC_VAL'
    upsert_env_line "V2_AUTO_MAX_LOTS" '$V2_AUTO_MAX_LOTS_VAL'
    upsert_env_line "V2_AUTO_REQUIRE_PREFERRED_BB" '$V2_AUTO_REQUIRE_PREFERRED_BB_VAL'
    upsert_env_line "V2_WATCH_REENTRY_COOLDOWN_SEC" '$V2_WATCH_REENTRY_COOLDOWN_SEC_VAL'
    upsert_env_line "V2_WATCH_MAX_DIRECTION_FLIPS" '$V2_WATCH_MAX_DIRECTION_FLIPS_VAL'
    upsert_env_line "INVALIDATION_SPOT_TRIGGER_BUFFER" '$INVALIDATION_SPOT_TRIGGER_BUFFER_VAL'
${STRATEGY_ENV_UPSERT_LINES}
EOF

# Copy Firebase service account JSON to EC2 (secret file; should NOT be in git)
if [ -n "$FCM_JSON_LOCAL_SOURCE_PATH" ] && [ -f "$FCM_JSON_LOCAL_SOURCE_PATH" ]; then
    echo "📤 Uploading Firebase service account JSON to EC2..."
    REMOTE_JSON_DIR="$REMOTE_API_PATH/$(dirname "$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE")"
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "mkdir -p \"$REMOTE_JSON_DIR\""
    scp -i "$PEM_FILE" "$FCM_JSON_LOCAL_SOURCE_PATH" "$EC2_USER@$EC2_IP:$REMOTE_API_PATH/$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE"
    echo "  ✓ Uploaded $FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE to $REMOTE_API_PATH/"
else
    echo "  ⚠️  Skipping Firebase service account JSON upload (missing local file or unset FCM_SERVICE_ACCOUNT_JSON)"
fi

# Sensex backtest data (gitignored under data/ — required for /sensex/backtest)
SENSEX_DATA_LOCAL="$REPO_ROOT/data/sensex"
SENSEX_DATA_REMOTE="$REMOTE_API_PATH/data/sensex"
if [ -f "$SENSEX_DATA_LOCAL/weekly_expiry_day_ohlc.csv" ]; then
    echo "📤 Uploading Sensex backtest session list..."
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "mkdir -p \"$SENSEX_DATA_REMOTE/dhan_intraday\""
    scp -i "$PEM_FILE" "$SENSEX_DATA_LOCAL/weekly_expiry_day_ohlc.csv" "$EC2_USER@$EC2_IP:$SENSEX_DATA_REMOTE/"
    if [ -d "$SENSEX_DATA_LOCAL/dhan_intraday" ] && [ "$(ls -A "$SENSEX_DATA_LOCAL/dhan_intraday" 2>/dev/null)" ]; then
        echo "📤 Uploading Dhan intraday cache ($(ls "$SENSEX_DATA_LOCAL/dhan_intraday" | wc -l | tr -d ' ') sessions)..."
        scp -i "$PEM_FILE" "$SENSEX_DATA_LOCAL/dhan_intraday/"*.json "$EC2_USER@$EC2_IP:$SENSEX_DATA_REMOTE/dhan_intraday/" 2>/dev/null || true
    fi
    echo "  ✓ Sensex backtest data synced"
else
    echo "  ⚠️  Sensex weekly_expiry_day_ohlc.csv missing locally — backtest UI will fail on EC2"
fi

# Nifty DuckDB + cached backtest JSON (gitignored — required for nifty50 Dhan viewer/backtest UI)
MARKET_DATA_LOCAL="$REPO_ROOT/data/market"
MARKET_DATA_REMOTE="$REMOTE_API_PATH/data/market"
if [ -d "$MARKET_DATA_LOCAL" ]; then
    echo "📤 Uploading Nifty market data + backtest results..."
    ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" "mkdir -p \"$MARKET_DATA_REMOTE\""
    for _f in \
        nifty50.staging.duckdb \
        nifty50.duckdb \
        backtest_nifty50_dhan_results.json \
        backtest_nifty50_sensex_trades.csv \
        tuned_20rupees_config.json; do
        if [ -f "$MARKET_DATA_LOCAL/$_f" ]; then
            echo "  → $_f"
            scp -i "$PEM_FILE" "$MARKET_DATA_LOCAL/$_f" "$EC2_USER@$EC2_IP:$MARKET_DATA_REMOTE/"
        fi
    done
    if [ -f "$SENSEX_DATA_LOCAL/backtest_20rupees_dhan_results.json" ]; then
        echo "  → sensex/backtest_20rupees_dhan_results.json"
        scp -i "$PEM_FILE" "$SENSEX_DATA_LOCAL/backtest_20rupees_dhan_results.json" "$EC2_USER@$EC2_IP:$SENSEX_DATA_REMOTE/../sensex/"
    fi
    echo "  ✓ Market/backtest data synced"
fi

# Now continue with the main deployment
echo ""
echo "📦 Starting main deployment process..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" << EOF
    set -e
    
    echo "📂 Navigating to API directory..."
    cd $REMOTE_API_PATH
    
    echo "📥 Syncing latest code from git..."
    # Stash tracked changes only — never stash algo-env (untracked venv; losing it forces a heavy pip reinstall).
    git stash push -m "deploy-pre-pull-\$(date +%Y%m%d%H%M%S)" 2>/dev/null || true

    # Remove untracked paths that block merge when stash did not catch them (e.g. partial copies on EC2).
    for f in \
        scripts/verify_fcm_service_account.py \
        services/v2_constants.py \
        services/v2_entry_pricing.py \
        services/v2_strategy_watch.py \
        api/v1/routes/v2_trading.py; do
        if [ -e "\$f" ] && ! git ls-files --error-unmatch -- "\$f" >/dev/null 2>&1; then
            if ! git check-ignore -q -- "\$f" 2>/dev/null; then
                echo "  🧹 Removing untracked \$f so pull can add the tracked version..."
                rm -f "\$f"
            fi
        fi
    done

    git fetch origin main
    git reset --hard origin/main

    # Drop most recent deploy autostash if present (server should match origin/main only).
    if git stash list 2>/dev/null | head -1 | grep -q 'deploy-pre-pull'; then
        git stash drop 2>/dev/null || true
    fi
    
    if [ ! -x "$REMOTE_API_PATH/algo-env/bin/python3.11" ]; then
        echo "❌ Virtualenv missing at $REMOTE_API_PATH/algo-env"
        echo "   Create once on EC2: python3.11 -m venv algo-env && algo-env/bin/pip install -r requirements.txt"
        exit 1
    fi

    if [ "${DEPLOY_INSTALL_DEPS:-0}" = "1" ]; then
        echo "📦 Installing Python dependencies (DEPLOY_INSTALL_DEPS=1)..."
        $REMOTE_API_PATH/algo-env/bin/python3.11 -m pip install -r requirements.txt
    else
        echo "⏭️  Skipping pip install (set DEPLOY_INSTALL_DEPS=1 to enable)"
    fi

    echo "🔄 Restarting API service..."
    sudo systemctl daemon-reload
    sudo systemctl restart $SERVICE_NAME
    
    echo "⏳ Waiting 3 seconds for service to start..."
    sleep 3
    
    echo "✅ Verifying service status..."
    sudo systemctl status $SERVICE_NAME --no-pager -l
    
    echo "📊 Checking service logs (last 20 lines)..."
    sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
    
    echo "✨ Backend deployment complete!"
EOF

echo "🎉 Deployment finished successfully!"
echo ""
echo "💡 To check service status manually, run:"
echo "   ssh -i $PEM_FILE $EC2_USER@$EC2_IP 'sudo systemctl status $SERVICE_NAME'"
echo ""
echo "💡 To view live logs, run:"
echo "   ssh -i $PEM_FILE $EC2_USER@$EC2_IP 'sudo journalctl -u $SERVICE_NAME -f'"


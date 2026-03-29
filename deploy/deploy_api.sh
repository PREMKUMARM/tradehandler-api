#!/bin/bash

# =================================================================
# AlgoFeast Backend Deployment Script
# This script connects to EC2, pulls latest code, updates dependencies,
# and restarts the API service.
# =================================================================

# --- CONFIGURATION ---
# Path to your .pem file (Change this to your actual path)
PEM_FILE="~/Downloads/algofeast-pkapps1993.pem"

# EC2 Connection Details
EC2_USER="ubuntu"
EC2_IP="13.233.151.3"

# Remote Paths
REMOTE_API_PATH="/home/ubuntu/algofeast-workspace/algofeast-api"
SERVICE_NAME="algofeast-api"
LOCAL_ENV_FILE=".env"
REMOTE_ENV_FILE="$REMOTE_API_PATH/.env"

# Binance-related environment variables to sync
BINANCE_VARS=("BINANCE_API_KEY" "BINANCE_API_SECRET" "BINANCE_SYMBOLS")

# --- EXECUTION ---

# Exit on any error
set -e

echo "🚀 Starting backend deployment to $EC2_IP..."

# Extract Binance + Telegram environment variables from local .env
echo "🔐 Extracting Binance & Telegram variables from local .env..."
BINANCE_API_KEY_VAL=""
BINANCE_API_SECRET_VAL=""
BINANCE_SYMBOLS_VAL=""
TELEGRAM_BOT_TOKEN_VAL=""
TELEGRAM_CHAT_ID_VAL=""

if [ -f "$LOCAL_ENV_FILE" ]; then
    BINANCE_API_KEY_VAL=$(grep "^BINANCE_API_KEY=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    BINANCE_API_SECRET_VAL=$(grep "^BINANCE_API_SECRET=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    BINANCE_SYMBOLS_VAL=$(grep "^BINANCE_SYMBOLS=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    TELEGRAM_BOT_TOKEN_VAL=$(grep "^TELEGRAM_BOT_TOKEN=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    TELEGRAM_CHAT_ID_VAL=$(grep "^TELEGRAM_CHAT_ID=" "$LOCAL_ENV_FILE" 2>/dev/null | cut -d '=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")
    
    [ -n "$BINANCE_API_KEY_VAL" ] && echo "  ✓ Found BINANCE_API_KEY"
    [ -n "$BINANCE_API_SECRET_VAL" ] && echo "  ✓ Found BINANCE_API_SECRET"
    [ -n "$BINANCE_SYMBOLS_VAL" ] && echo "  ✓ Found BINANCE_SYMBOLS"
    [ -n "$TELEGRAM_BOT_TOKEN_VAL" ] && echo "  ✓ Found TELEGRAM_BOT_TOKEN"
    [ -n "$TELEGRAM_CHAT_ID_VAL" ] && echo "  ✓ Found TELEGRAM_CHAT_ID"
else
    echo "  ⚠️  Local .env file not found, skipping environment variable sync"
fi

# Connect to EC2 and sync environment variables first
echo "🔐 Syncing Binance & Telegram environment variables to EC2..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" bash << EOF
    set -e
    
    REMOTE_ENV_FILE="$REMOTE_ENV_FILE"
    BINANCE_API_KEY_VAL='$BINANCE_API_KEY_VAL'
    BINANCE_API_SECRET_VAL='$BINANCE_API_SECRET_VAL'
    BINANCE_SYMBOLS_VAL='$BINANCE_SYMBOLS_VAL'
    TELEGRAM_BOT_TOKEN_VAL='$TELEGRAM_BOT_TOKEN_VAL'
    TELEGRAM_CHAT_ID_VAL='$TELEGRAM_CHAT_ID_VAL'
    
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
EOF

# Now continue with the main deployment
echo ""
echo "📦 Starting main deployment process..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" << EOF
    set -e
    
    echo "📂 Navigating to API directory..."
    cd $REMOTE_API_PATH
    
    echo "📥 Pulling latest code from git..."
    git stash
    git pull
    
    echo "🔧 Activating virtual environment..."
    source algo-env/bin/activate
    
    echo "📦 Installing/updating Python dependencies..."
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    
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


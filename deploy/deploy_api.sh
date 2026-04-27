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

# Normalize FCM JSON path for EC2:
# - If your local .env uses an absolute path, we still deploy the file into the API directory
#   and write FCM_SERVICE_ACCOUNT_JSON as a repo-relative path (basename) on the server.
FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE="$FCM_SERVICE_ACCOUNT_JSON_VAL"
if [[ "$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE" = /* ]]; then
    FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE="$(basename "$FCM_SERVICE_ACCOUNT_JSON_FOR_REMOTE")"
fi

# If FCM JSON is a relative path, resolve it from repo root (this script lives in deploy/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# Now continue with the main deployment
echo ""
echo "📦 Starting main deployment process..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" << EOF
    set -e
    
    echo "📂 Navigating to API directory..."
    cd $REMOTE_API_PATH
    
    echo "📥 Pulling latest code from git..."
    git stash
    # If someone copied a file onto the server without "git add", the same path may exist in
    # a newer commit; git then aborts: "untracked working tree files would be overwritten".
    # Remove *untracked* (not in index) non-ignored paths we know the repo now tracks.
    for f in scripts/verify_fcm_service_account.py; do
        if [ -e "\$f" ] && ! git ls-files --error-unmatch -- "\$f" >/dev/null 2>&1; then
            if ! git check-ignore -q -- "\$f" 2>/dev/null; then
                echo "  🧹 Removing untracked \$f so pull can add the tracked version..."
                rm -f "\$f"
            fi
        fi
    done
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


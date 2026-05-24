#!/bin/bash

# =================================================================
# vibeFnO Frontend Build & Deploy Script
# Builds Angular locally, uploads to EC2, and serves via nginx on :80
# =================================================================

# --- CONFIGURATION ---
PEM_FILE="/Users/premkumar/Documents/vibefno.pem"

EC2_USER="ec2-user"
EC2_IP="ec2-3-108-61-102.ap-south-1.compute.amazonaws.com"

LOCAL_PROJECT_ROOT="/Users/premkumar/Documents/tradehandler-ai-workspace/tradehandler"
DIST_PATH="$LOCAL_PROJECT_ROOT/dist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_CONF="$SCRIPT_DIR/nginx.conf"

REMOTE_STAGING="/home/ec2-user/vibefno-deploy/staging"
REMOTE_WEB_ROOT="/var/www/vibefno"
REMOTE_NGINX_CONF="/etc/nginx/conf.d/vibefno.conf"
DOMAIN="${DOMAIN:-vibefno.com}"

# --- EXECUTION ---
set -e

echo "🚀 Starting vibeFnO UI deployment to $EC2_IP..."

echo "📦 Building Angular app locally (production)..."
cd "$LOCAL_PROJECT_ROOT"
export NODE_OPTIONS="--max-old-space-size=4096"

if npm run build; then
    echo "✅ Local build successful."
else
    echo "❌ Local build failed. Aborting."
    exit 1
fi

echo "🗜️ Compressing build files..."
cd "$DIST_PATH"
tar -czf ../dist.tar.gz .
cd "$LOCAL_PROJECT_ROOT"

echo "☁️ Uploading build and nginx config to EC2..."
scp -i "$PEM_FILE" dist.tar.gz "$NGINX_CONF" "$EC2_USER@$EC2_IP:/tmp/"

echo "🔧 Installing nginx (if needed) and deploying UI on port 80..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" bash << EOF
    set -e

    if ! command -v nginx >/dev/null 2>&1; then
        echo "📥 Installing nginx..."
        sudo dnf install -y nginx
    fi

    echo "📁 Preparing web root..."
    sudo mkdir -p "$REMOTE_WEB_ROOT"
    mkdir -p "$REMOTE_STAGING"
    tar -xzf /tmp/dist.tar.gz -C "$REMOTE_STAGING"

    echo "🔄 Updating nginx site..."
    sudo rm -f /etc/nginx/conf.d/default.conf
    if ! grep -q 'server_names_hash_bucket_size' /etc/nginx/nginx.conf; then
        sudo sed -i '/http {/a \    server_names_hash_bucket_size 128;' /etc/nginx/nginx.conf
    fi
    sudo rm -rf "$REMOTE_WEB_ROOT"/*
    sudo cp -r "$REMOTE_STAGING"/* "$REMOTE_WEB_ROOT"/
    if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
        sudo cp /tmp/nginx.conf "$REMOTE_NGINX_CONF"
    else
        echo "🔒 SSL active for $DOMAIN — keeping certbot nginx config, updating files only."
    fi
    sudo rm -f /etc/nginx/conf.d/default.conf

    sudo chown -R nginx:nginx "$REMOTE_WEB_ROOT"
    sudo chmod -R 755 "$REMOTE_WEB_ROOT"

    echo "✅ Testing nginx configuration..."
    sudo nginx -t

    sudo systemctl enable nginx
    sudo systemctl restart nginx

    rm -f /tmp/dist.tar.gz /tmp/nginx.conf
    rm -rf "$REMOTE_STAGING"

    echo "✨ Remote update complete!"
    curl -s -o /dev/null -w "Local nginx HTTP %{http_code}\n" http://127.0.0.1/
EOF

rm -f "$LOCAL_PROJECT_ROOT/dist.tar.gz"

echo "🎉 Deployment finished successfully!"
echo ""
echo "🌐 App URL: http://$EC2_IP/"
echo "💡 Ensure EC2 security group allows inbound TCP port 80."

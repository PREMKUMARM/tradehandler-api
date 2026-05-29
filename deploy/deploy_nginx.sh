#!/bin/bash
# Deploy nginx WebSocket config to EC2 (preserves Certbot SSL when present).

set -e

PEM_FILE="/Users/premkumar/Documents/vibefno.pem"
EC2_USER="ec2-user"
EC2_IP="ec2-3-108-61-102.ap-south-1.compute.amazonaws.com"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_NGINX_CONF="/etc/nginx/conf.d/vibefno.conf"
DOMAIN="${DOMAIN:-vibefno.com}"

echo "🚀 Deploying nginx config to $EC2_IP..."

scp -i "$PEM_FILE" \
    "$SCRIPT_DIR/nginx.conf" \
    "$SCRIPT_DIR/vibefno-ssl.conf" \
    "$EC2_USER@$EC2_IP:/tmp/"

ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" bash << EOF
    set -e
    DOMAIN="$DOMAIN"
    REMOTE_NGINX_CONF="$REMOTE_NGINX_CONF"

    if ! command -v nginx >/dev/null 2>&1; then
        sudo dnf install -y nginx
    fi

    sudo rm -f /etc/nginx/conf.d/default.conf

    if sudo test -d "/etc/letsencrypt/live/\$DOMAIN"; then
        echo "🔒 SSL detected — installing vibefno-ssl.conf"
        sudo cp /tmp/vibefno-ssl.conf "\$REMOTE_NGINX_CONF"
    else
        echo "📄 No SSL — installing HTTP-only nginx.conf"
        sudo cp /tmp/nginx.conf "\$REMOTE_NGINX_CONF"
    fi

    echo "✅ Testing nginx..."
    sudo nginx -t
    sudo systemctl enable nginx
    sudo systemctl reload nginx
    rm -f /tmp/nginx.conf /tmp/vibefno-ssl.conf
    echo "✨ nginx reloaded"
EOF

echo "🎉 nginx deploy complete"

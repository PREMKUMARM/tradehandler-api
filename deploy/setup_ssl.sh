#!/bin/bash

# =================================================================
# vibeFnO SSL Setup (Let's Encrypt via Certbot)
# Requires DNS A records for DOMAIN -> EC2 public IP before running.
# =================================================================

set -e

DOMAIN="${DOMAIN:-vibefno.com}"
SSL_EMAIL="${SSL_EMAIL:-admin@vibefno.com}"
EC2_PUBLIC_IP="${EC2_PUBLIC_IP:-3.108.61.102}"

PEM_FILE="/Users/premkumar/Documents/vibefno.pem"
EC2_USER="ec2-user"
EC2_IP="ec2-3-108-61-102.ap-south-1.compute.amazonaws.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_CONF="$SCRIPT_DIR/nginx.conf"
REMOTE_API_ENV="/home/ec2-user/algofeast-workspace/algofeast-api/.env"
REMOTE_NGINX_CONF="/etc/nginx/conf.d/vibefno.conf"

echo "🔒 Setting up SSL for $DOMAIN and www.$DOMAIN on $EC2_IP..."

echo "🌐 Checking DNS..."
RESOLVED=$(dig +short "$DOMAIN" A | tail -1)
WWW_RESOLVED=$(dig +short "www.$DOMAIN" A | tail -1)
if [ "$RESOLVED" != "$EC2_PUBLIC_IP" ] && [ "$WWW_RESOLVED" != "$EC2_PUBLIC_IP" ]; then
    echo "⚠️  DNS is not pointing to $EC2_PUBLIC_IP yet."
    echo "   $DOMAIN -> ${RESOLVED:-<none>}"
    echo "   www.$DOMAIN -> ${WWW_RESOLVED:-<none>}"
    echo ""
    echo "   Add these DNS records at your registrar, then re-run:"
    echo "   A    $DOMAIN      -> $EC2_PUBLIC_IP"
    echo "   A    www.$DOMAIN  -> $EC2_PUBLIC_IP"
    echo ""
    echo "   Also open EC2 security group inbound TCP 443."
    exit 1
fi

echo "☁️ Uploading nginx config..."
scp -i "$PEM_FILE" "$NGINX_CONF" "$EC2_USER@$EC2_IP:/tmp/nginx.conf"

ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" bash << EOF
    set -e
    DOMAIN="$DOMAIN"
    SSL_EMAIL="$SSL_EMAIL"
    EC2_PUBLIC_IP="$EC2_PUBLIC_IP"
    REMOTE_API_ENV="$REMOTE_API_ENV"
    REMOTE_NGINX_CONF="$REMOTE_NGINX_CONF"

    if ! command -v nginx >/dev/null 2>&1; then
        sudo dnf install -y nginx
    fi

    if ! command -v certbot >/dev/null 2>&1; then
        echo "📥 Installing certbot..."
        sudo dnf install -y certbot python3-certbot-nginx
    fi

    sudo rm -f /etc/nginx/conf.d/default.conf
    if ! grep -q 'server_names_hash_bucket_size' /etc/nginx/nginx.conf; then
        sudo sed -i '/http {/a \    server_names_hash_bucket_size 128;' /etc/nginx/nginx.conf
    fi

    sudo cp /tmp/nginx.conf "\$REMOTE_NGINX_CONF"
    sudo nginx -t
    sudo systemctl enable nginx
    sudo systemctl restart nginx

    echo "🔐 Requesting Let's Encrypt certificate..."
    sudo certbot --nginx \
        -d "\$DOMAIN" \
        -d "www.\$DOMAIN" \
        --non-interactive \
        --agree-tos \
        -m "\$SSL_EMAIL" \
        --redirect

    upsert_env_line() {
        local key="\$1"
        local val="\$2"
        local f="\$REMOTE_API_ENV"
        [ -z "\$val" ] && return 0
        touch "\$f"
        if grep -q "^\${key}=" "\$f" 2>/dev/null; then
            grep -v "^\${key}=" "\$f" > "\${f}.new" || true
            mv "\${f}.new" "\$f"
        fi
        echo "\${key}=\${val}" >> "\$f"
    }

    echo "🔧 Updating API CORS and Kite redirect for HTTPS..."
    upsert_env_line "CORS_ORIGINS" "https://\$DOMAIN,https://www.\$DOMAIN,http://$EC2_PUBLIC_IP"
  upsert_env_line "KITE_REDIRECT_URI" "https://\$DOMAIN/auth-token"
  upsert_env_line "ENVIRONMENT" "production"

  sudo systemctl restart algofeast-api

  echo "✅ SSL setup complete."
  curl -s -o /dev/null -w "HTTPS local check %{http_code}\n" "https://\$DOMAIN/" || true
EOF

echo ""
echo "🎉 HTTPS enabled!"
echo "   https://$DOMAIN/"
echo "   https://www.$DOMAIN/"

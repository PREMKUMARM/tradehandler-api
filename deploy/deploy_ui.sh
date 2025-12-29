#!/bin/bash

# =================================================================
# AlgoFeast Local Build & Deploy Script
# This script builds the Angular app locally and uploads it to EC2.
# =================================================================

# --- CONFIGURATION ---
# Path to your .pem file (Change this to your actual path)
PEM_FILE="~/Downloads/algofeast-pkapps1993.pem"

# EC2 Connection Details
EC2_USER="ubuntu"
EC2_IP="13.233.151.3"

# Local Paths
LOCAL_PROJECT_ROOT="/Users/premkumar/Documents/tradehandler-ai-workspace/tradehandler"
DIST_PATH="$LOCAL_PROJECT_ROOT/dist"

# Remote Paths
REMOTE_TEMP_PATH="/home/ubuntu/algofeast-workspace/algofeast-frontend/dist"
REMOTE_NGINX_PATH="/var/www/algofeast"

# --- EXECUTION ---

# Exit on any error
set -e

echo "🚀 Starting deployment to $EC2_IP..."

# 1. Build Locally
echo "📦 Building Angular app locally..."
cd "$LOCAL_PROJECT_ROOT"

# Increase memory limit for local build to be safe
export NODE_OPTIONS="--max-old-space-size=4096"

if npm run build; then
    echo "✅ Local build successful."
else
    echo "❌ Local build failed. Aborting."
    exit 1
fi

# 2. Compress the dist folder for faster transfer
echo "🗜️ Compressing build files..."
cd "$DIST_PATH"
tar -czf ../dist.tar.gz .
cd ..

# 3. Upload to EC2
echo "☁️ Uploading to EC2..."
scp -i "$PEM_FILE" dist.tar.gz "$EC2_USER@$EC2_IP:$REMOTE_TEMP_PATH.tar.gz"

# 4. Extract and move to Nginx folder on EC2
echo "🔧 Extracting files on EC2 and updating Nginx folder..."
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_IP" << EOF
    # Create temp dir if not exists
    mkdir -p $REMOTE_TEMP_PATH
    
    # Extract
    tar -xzf $REMOTE_TEMP_PATH.tar.gz -C $REMOTE_TEMP_PATH
    
    # Clear and move to production folder
    sudo rm -rf $REMOTE_NGINX_PATH/*
    sudo mv $REMOTE_TEMP_PATH/* $REMOTE_NGINX_PATH/
    
    # Cleanup
    rm $REMOTE_TEMP_PATH.tar.gz
    rm -rf $REMOTE_TEMP_PATH
    
    # Fix permissions
    sudo chown -R www-data:www-data $REMOTE_NGINX_PATH
    
    echo "✨ Remote update complete!"
EOF

# 5. Cleanup local tarball
rm dist.tar.gz

echo "🎉 Deployment finished successfully!"


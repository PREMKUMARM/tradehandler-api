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

# --- EXECUTION ---

# Exit on any error
set -e

echo "🚀 Starting backend deployment to $EC2_IP..."

# Connect to EC2 and deploy
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


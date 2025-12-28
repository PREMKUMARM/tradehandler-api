#!/bin/bash

# EC2 Setup Script for AlgoFeast
# Run this on your Ubuntu EC2 instance

set -e

echo "Starting EC2 Setup..."

# 1. Update and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx git curl

# 2. Install Node.js 20 (LTS)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 3. Create Swap Space (for 1GB RAM instances)
if [ ! -f /swapfile ]; then
    echo "Creating 2GB Swap file..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 4. Setup Backend
echo "Setting up backend..."
cd /home/ubuntu/algofeast-workspace/algofeast-api
python3 -m venv algo-env
source algo-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Setup Frontend
echo "Setting up frontend..."
cd /home/ubuntu/algofeast-workspace/algofeast-frontend
npm install
npm run build --configuration production
sudo mkdir -p /var/www/algofeast
sudo cp -r dist/* /var/www/algofeast/

# 6. Configure Systemd Service
echo "Configuring systemd..."
sudo cp /home/ubuntu/algofeast-workspace/algofeast-api/deploy/algofeast-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable algofeast-api
# Note: You need to create .env before starting
# sudo systemctl start algofeast-api

# 7. Configure Nginx
echo "Configuring Nginx..."
sudo cp /home/ubuntu/algofeast-workspace/algofeast-api/deploy/nginx.conf /etc/nginx/sites-available/algofeast
sudo ln -sf /etc/nginx/sites-available/algofeast /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

echo "Setup complete! Remember to:"
echo "1. Configure your .env file in /home/ubuntu/algofeast-workspace/algofeast-api"
echo "2. Run 'sudo certbot --nginx' for SSL"
echo "3. Restart services: sudo systemctl restart nginx algofeast-api"


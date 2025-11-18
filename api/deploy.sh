#!/bin/bash

# ============================================================================
# Medical PLS API - Deployment Script for AWS EC2
# ============================================================================

set -e  # Exit on error

echo "🚀 Medical PLS API - Deployment Script"
echo "========================================"

# Configuration
APP_DIR="/opt/pls-api"
API_DIR="$APP_DIR/api"
VENV_DIR="$API_DIR/venv"
SERVICE_NAME="pls-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}❌ Please do not run as root. Use a user with sudo privileges.${NC}"
    exit 1
fi

# Step 1: Update system
echo -e "\n${YELLOW}📦 Step 1: Updating system...${NC}"
sudo apt update
sudo apt upgrade -y

# Step 2: Install dependencies
echo -e "\n${YELLOW}📦 Step 2: Installing system dependencies...${NC}"
sudo apt install -y python3.9 python3.9-venv python3-pip build-essential git curl nginx

# Step 3: Create application directory
echo -e "\n${YELLOW}📁 Step 3: Setting up application directory...${NC}"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Step 4: Clone or update repository
if [ -d "$API_DIR" ]; then
    echo -e "${YELLOW}Repository exists, updating...${NC}"
    cd $API_DIR
    git pull || echo "Git pull failed, continuing..."
else
    echo -e "${YELLOW}Please clone your repository to $API_DIR${NC}"
    echo "Example: git clone YOUR_REPO_URL $APP_DIR"
    read -p "Press Enter after cloning repository..."
fi

cd $API_DIR

# Step 5: Create virtual environment
echo -e "\n${YELLOW}🐍 Step 5: Setting up Python virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3.9 -m venv venv
fi

source venv/bin/activate

# Step 6: Install Python dependencies
echo -e "\n${YELLOW}📦 Step 6: Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Step 7: Check model
echo -e "\n${YELLOW}🤖 Step 7: Checking model...${NC}"
if [ ! -d "$API_DIR/models/t5_base" ]; then
    echo -e "${RED}⚠️  Model not found at $API_DIR/models/t5_base${NC}"
    echo "Please ensure the model is in the correct location."
    echo "You can:"
    echo "  1. Upload via SCP: scp -r models/t5_base user@server:$API_DIR/models/"
    echo "  2. Download from S3: aws s3 cp s3://bucket/models/t5_base $API_DIR/models/t5_base --recursive"
    read -p "Press Enter after model is in place..."
fi

# Step 8: Create systemd service
echo -e "\n${YELLOW}⚙️  Step 8: Creating systemd service...${NC}"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Medical PLS API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$API_DIR
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Step 9: Enable and start service
echo -e "\n${YELLOW}🚀 Step 9: Starting service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

# Wait a moment for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✅ Service is running${NC}"
else
    echo -e "${RED}❌ Service failed to start. Check logs: sudo journalctl -u $SERVICE_NAME${NC}"
    exit 1
fi

# Step 10: Configure Nginx
echo -e "\n${YELLOW}🌐 Step 10: Configuring Nginx...${NC}"
read -p "Enter your domain or IP address (or press Enter to use IP): " DOMAIN
DOMAIN=${DOMAIN:-$(curl -s ifconfig.me)}

sudo tee /etc/nginx/sites-available/$SERVICE_NAME > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /static/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_cache_valid 200 1h;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # Remove default if exists

# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx

# Step 11: Test deployment
echo -e "\n${YELLOW}🧪 Step 11: Testing deployment...${NC}"
sleep 2

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is responding${NC}"
else
    echo -e "${RED}❌ API is not responding. Check logs: sudo journalctl -u $SERVICE_NAME${NC}"
fi

# Final instructions
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Your API is available at:"
echo "  - http://$DOMAIN"
echo "  - http://$DOMAIN/docs (API documentation)"
echo ""
echo "Useful commands:"
echo "  - View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  - Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  - Check status: sudo systemctl status $SERVICE_NAME"
echo ""
echo "Next steps:"
echo "  1. Configure SSL with Let's Encrypt: sudo certbot --nginx -d $DOMAIN"
echo "  2. Set up firewall: sudo ufw allow 80,443/tcp"
echo "  3. Monitor logs: sudo journalctl -u $SERVICE_NAME -f"
echo ""


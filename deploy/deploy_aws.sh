#!/bin/bash
# Deployment script for AWS EC2
# Deploy PLS Generator Dashboard to AWS EC2

echo "🚀 Deploying PLS Generator to AWS EC2..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
echo -e "${BLUE}📦 Installing dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# ============================================================================
# 2. CREATE VIRTUAL ENVIRONMENT
# ============================================================================
echo -e "${BLUE}🐍 Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# ============================================================================
# 3. INSTALL PYTHON PACKAGES
# ============================================================================
echo -e "${BLUE}📚 Installing Python packages...${NC}"
pip install --upgrade pip
pip install streamlit
pip install torch transformers accelerate
pip install pandas numpy
pip install --upgrade streamlit

# ============================================================================
# 4. SETUP FIREWALL (Security Group)
# ============================================================================
echo -e "${BLUE}🔥 Configuring firewall...${NC}"
# Note: Configure Security Group in AWS Console:
# - Allow inbound traffic on port 8501 (Streamlit default)

# ============================================================================
# 5. CREATE SYSTEMD SERVICE
# ============================================================================
echo -e "${BLUE}⚙️  Creating systemd service...${NC}"
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit PLS Generator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
Environment="PATH=/home/$USER/venv/bin"
ExecStart=/home/$USER/venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ============================================================================
# 6. START SERVICE
# ============================================================================
echo -e "${BLUE}🔄 Starting Streamlit service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit

# ============================================================================
# 7. CHECK STATUS
# ============================================================================
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Status:"
sudo systemctl status streamlit

echo ""
echo -e "${GREEN}Access your dashboard at:${NC}"
echo "http://$(curl -s http://checkip.amazonaws.com):8501"

echo ""
echo "Management commands:"
echo "  sudo systemctl status streamlit  # Check status"
echo "  sudo systemctl restart streamlit # Restart"
echo "  sudo systemctl stop streamlit    # Stop"
echo "  sudo systemctl start streamlit   # Start"



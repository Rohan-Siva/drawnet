#!/bin/bash

# DrawNet Setup Script
# Automates installation and initial setup

set -e  # Exit on error

echo "============================================================"
echo "DrawNet Setup"
echo "============================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python $python_version"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "  ✓ Virtual environment created"
    echo ""
    echo "To activate:"
    echo "  source venv/bin/activate  # Mac/Linux"
    echo "  venv\\Scripts\\activate     # Windows"
    echo ""
    read -p "Press enter to continue..."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Running installation test..."
echo "============================================================"
python test_installation.py

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Add images to data/raw/"
echo "  2. Run: python prepare_data.py --process"
echo "  3. Run: python train.py"
echo ""
echo "For help, see README.md"
echo "============================================================"

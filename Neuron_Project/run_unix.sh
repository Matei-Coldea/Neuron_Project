#!/bin/bash
# Unix launcher script for 3D TIFF Viewer (Linux/macOS)
# Make executable with: chmod +x run_unix.sh

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}3D TIFF Viewer - Unix Launcher${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=macOS;;
    *)          MACHINE="UNKNOWN:${OS}"
esac
echo -e "${GREEN}[*]${NC} Detected platform: ${MACHINE}"

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo -e "${GREEN}[✓]${NC} Virtual environment found"
    echo -e "${YELLOW}[*]${NC} Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "${RED}[!]${NC} Virtual environment not found"
    echo -e "${RED}[!]${NC} Please create one first with: python3 -m venv venv"
    echo -e "${RED}[!]${NC} Then install dependencies: pip install -r requirements.txt"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Python version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}[✓]${NC} Python found: ${PYTHON_VERSION}"
else
    echo -e "${RED}[✗]${NC} Python not found in PATH"
    echo -e "${RED}[!]${NC} Please install Python 3.8 or higher"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check for required packages
echo -e "${YELLOW}[*]${NC} Checking for required packages..."
if ! python -c "import PySide6" 2>/dev/null; then
    echo -e "${YELLOW}[!]${NC} PySide6 not installed"
    echo -e "${YELLOW}[*]${NC} Installing dependencies..."
    pip install -r requirements.txt
fi

# Platform-specific checks
if [ "${MACHINE}" = "Linux" ]; then
    echo ""
    echo -e "${CYAN}Linux-specific checks:${NC}"
    
    # Check for OpenGL libraries
    if ! ldconfig -p | grep -q libGL.so; then
        echo -e "${YELLOW}[!]${NC} OpenGL libraries may not be installed"
        echo -e "${YELLOW}[!]${NC} Try: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
    else
        echo -e "${GREEN}[✓]${NC} OpenGL libraries found"
    fi
    
    # Check display server
    if [ -n "$DISPLAY" ]; then
        echo -e "${GREEN}[✓]${NC} X11 display available: $DISPLAY"
    elif [ -n "$WAYLAND_DISPLAY" ]; then
        echo -e "${YELLOW}[!]${NC} Wayland detected - may need XWayland"
    else
        echo -e "${RED}[!]${NC} No display server detected"
    fi
    
elif [ "${MACHINE}" = "macOS" ]; then
    echo ""
    echo -e "${CYAN}macOS-specific checks:${NC}"
    
    # Check for Xcode Command Line Tools
    if xcode-select -p &> /dev/null; then
        echo -e "${GREEN}[✓]${NC} Xcode Command Line Tools installed"
    else
        echo -e "${YELLOW}[!]${NC} Xcode Command Line Tools not found"
        echo -e "${YELLOW}[!]${NC} Install with: xcode-select --install"
    fi
fi

echo ""
echo -e "${YELLOW}[*]${NC} Launching 3D TIFF Viewer..."
echo ""

# Run the main application
python main.py
EXIT_CODE=$?

# Check exit code
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}[✗]${NC} Application exited with error code: ${EXIT_CODE}"
    echo ""
    echo -e "${CYAN}Troubleshooting tips:${NC}"
    
    if [ "${MACHINE}" = "Linux" ]; then
        echo "1. Install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
        echo "2. Check OpenGL: glxinfo | grep 'OpenGL version'"
        echo "3. Update Mesa drivers"
    elif [ "${MACHINE}" = "macOS" ]; then
        echo "1. Install Xcode Command Line Tools: xcode-select --install"
        echo "2. Try: pip install --upgrade --force-reinstall PySide6"
        echo "3. Check System Preferences → Security & Privacy"
    fi
    
    echo "General: pip install -r requirements.txt --force-reinstall"
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo -e "${GREEN}[✓]${NC} Application closed successfully"


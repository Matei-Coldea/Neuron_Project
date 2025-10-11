@echo off
REM Windows launcher script for 3D TIFF Viewer with AUTOMATIC FIX CAPABILITIES
REM This script handles common Windows issues automatically

setlocal EnableDelayedExpansion

echo ========================================
echo 3D TIFF Viewer - Windows Launcher v2.0
echo ========================================
echo.

REM Check if running with administrator privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [^!] Running with Administrator privileges
) else (
    echo [^!] NOT running as Administrator
    echo [^!] Some automatic fixes may not work
    echo [^!] Consider right-clicking and "Run as administrator"
    echo.
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [✓] Virtual environment found
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [✗] Virtual environment not found
    echo.
    echo [*] Creating virtual environment automatically...
    python -m venv venv
    if errorlevel 1 (
        echo [✗] Failed to create virtual environment
        echo [!] Please install Python 3.8 or higher
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
    call venv\Scripts\activate.bat
    
    echo [*] Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [✗] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [✓] Dependencies installed
)

REM Check Python version
echo [*] Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo [✗] Python not found in PATH
    echo [!] Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Check and install missing critical packages
echo [*] Checking critical packages...

python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo [!] PySide6 not installed - installing now...
    pip install PySide6 PySide6-Addons PySide6-Essentials
    if errorlevel 1 (
        echo [✗] Failed to install PySide6
        echo [*] Trying alternative installation method...
        pip install --upgrade --force-reinstall PySide6
    )
)

python -c "import OpenGL" 2>nul
if errorlevel 1 (
    echo [!] PyOpenGL not installed - installing now...
    pip install PyOpenGL PyOpenGL-accelerate
)

python -c "import numpy" 2>nul
if errorlevel 1 (
    echo [!] NumPy not installed - installing now...
    pip install numpy
)

REM Check for binary incompatibility issues (numpy vs scikit-image)
python -c "import skimage" 2>nul
if errorlevel 1 (
    echo [!] scikit-image has issues - reinstalling compatible version...
    pip install --upgrade numpy scikit-image --no-cache-dir
)

echo [✓] All critical packages available
echo.

REM Set Windows-specific environment variables for better compatibility
echo [*] Configuring Windows environment...
set MPLBACKEND=Agg
set QT_QPA_PLATFORM=windows
set QT_OPENGL=desktop
set LIBGL_ALWAYS_SOFTWARE=0
set PYTHONUNBUFFERED=1
echo [✓] Environment configured

REM Display GPU information
echo.
echo [*] Detecting GPU...
wmic path win32_VideoController get name 2>nul | findstr /v "Name" | findstr /v "^$"
if errorlevel 1 (
    echo [!] Could not detect GPU
) else (
    REM Check if only Intel GPU is present (common issue)
    wmic path win32_VideoController get name | findstr /i "Intel" >nul
    if not errorlevel 1 (
        wmic path win32_VideoController get name | findstr /i "NVIDIA AMD Radeon GeForce" >nul
        if errorlevel 1 (
            echo.
            echo [^!^!] WARNING: Only Intel integrated graphics detected
            echo [^!^!] This may cause performance issues or crashes
            echo [^!^!] If you have a dedicated GPU, configure it in:
            echo [^!^!] Settings -^> System -^> Display -^> Graphics settings
            echo [^!^!] Add python.exe and set to "High performance"
            echo.
            timeout /t 3 >nul
        )
    )
)

echo.
echo ========================================
echo Launching 3D TIFF Viewer...
echo ========================================
echo.
echo Controls:
echo - Left mouse button: Rotate view
echo - Middle mouse button: Pan view
echo - Scroll wheel: Zoom in/out
echo.
echo Common Windows issues:
echo - If crash on startup: Update graphics drivers
echo - If blank window: Try running as Administrator
echo - If slow performance: Lower Max Points in settings
echo.

REM Launch the application
python main.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ========================================
    echo Application exited with an error
    echo ========================================
    echo.
    
    REM Try to diagnose the issue
    echo [*] Running diagnostics...
    echo.
    
    REM Check OpenGL availability
    python -c "from OpenGL.GL import glGetString, GL_VERSION; print('OpenGL test passed')" 2>nul
    if errorlevel 1 (
        echo [✗] OpenGL test FAILED
        echo.
        echo This usually means:
        echo 1. Graphics drivers are outdated or missing
        echo 2. OpenGL is not properly installed
        echo.
        echo AUTOMATIC FIX ATTEMPT:
        echo [*] Reinstalling PyOpenGL...
        pip install --upgrade --force-reinstall PyOpenGL PyOpenGL-accelerate
        echo.
        echo Please also:
        echo 1. Update your graphics drivers from manufacturer:
        echo    - Intel: https://www.intel.com/content/www/us/en/download-center/
        echo    - NVIDIA: https://www.nvidia.com/Download/index.aspx
        echo    - AMD: https://www.amd.com/en/support
        echo 2. Restart your computer after updating drivers
        echo 3. Run this script again as Administrator
    ) else (
        echo [✓] OpenGL test passed
        echo.
        echo The error may be related to:
        echo 1. Qt platform plugins
        echo 2. Display scaling settings
        echo 3. Antivirus/firewall interference
        echo.
        echo AUTOMATIC FIX ATTEMPT:
        echo [*] Reinstalling PySide6...
        pip install --upgrade --force-reinstall PySide6
        echo.
        echo Please also try:
        echo 1. Right-click python.exe in venv\Scripts\
        echo 2. Properties -^> Compatibility tab
        echo 3. Check "Override high DPI scaling behavior"
        echo 4. Select "Application" from dropdown
    )
    
    echo.
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo [✓] Application closed successfully
echo ========================================
pause

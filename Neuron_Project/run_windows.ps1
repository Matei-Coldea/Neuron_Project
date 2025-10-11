# PowerShell launcher script for 3D TIFF Viewer
# Run with: powershell -ExecutionPolicy Bypass -File run_windows.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3D TIFF Viewer - Windows PowerShell Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[✓] Virtual environment found" -ForegroundColor Green
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "[!] Virtual environment not found" -ForegroundColor Red
    Write-Host "[!] Please create one first with: python -m venv venv" -ForegroundColor Red
    Write-Host "[!] Then install dependencies: pip install -r requirements.txt" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Python not found in PATH" -ForegroundColor Red
    Write-Host "[!] Please install Python 3.8 or higher" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for required packages
Write-Host "[*] Checking for required packages..." -ForegroundColor Yellow
try {
    python -c "import PySide6" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] PySide6 not installed" -ForegroundColor Yellow
        Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
} catch {
    Write-Host "[!] Error checking dependencies" -ForegroundColor Red
}

Write-Host ""
Write-Host "[*] Launching 3D TIFF Viewer..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Common Windows issues:" -ForegroundColor Cyan
Write-Host "- If the app crashes immediately, update your graphics drivers" -ForegroundColor Gray
Write-Host "- If dual GPU: Set Python to use dedicated GPU in Windows Settings" -ForegroundColor Gray
Write-Host "- If display is blank: Try disabling display scaling for Python.exe" -ForegroundColor Gray
Write-Host ""

# Run the main application
python main.py

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[✗] Application exited with an error" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting tips:" -ForegroundColor Cyan
    Write-Host "1. Update graphics drivers (Intel/NVIDIA/AMD)" -ForegroundColor Gray
    Write-Host "2. Run PowerShell as Administrator" -ForegroundColor Gray
    Write-Host "3. Try: pip install --upgrade --force-reinstall PySide6" -ForegroundColor Gray
    Write-Host "4. Check Windows Update for driver updates" -ForegroundColor Gray
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[✓] Application closed successfully" -ForegroundColor Green
Read-Host "Press Enter to exit"


# VS Code Debug Mode Fix for Windows

## The Problem

When running `main.py` in VS Code debug mode on Windows, the application may:
- Fail to start
- Show matplotlib errors
- Crash with Qt/OpenGL errors
- Not load environment variables properly

## Why Debug Mode is Different

VS Code debugger (debugpy):
1. **Instruments your code** before running it
2. **Imports modules** during setup phase
3. **May not load** `.env` file before code runs
4. **Attaches** to Python process after some imports
5. **Different import order** than normal execution

This causes issues with:
- matplotlib trying to choose Qt backend during debugger setup
- PySide6 not being initialized when matplotlib loads
- Environment variables not being set in time

## The Solution

We've implemented a **3-layer fix**:

### Layer 1: Module-Level Environment Setup (Highest Priority)

In `main.py`, we set environment variables **immediately** at module level:

```python
import os
import platform

# Set BEFORE any other imports
os.environ.setdefault('MPLBACKEND', 'Agg')
if platform.system() == 'Windows':
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
    os.environ.setdefault('QT_OPENGL', 'desktop')
```

This runs **before** the VS Code debugger can instrument any imports.

### Layer 2: VS Code Configuration Files

**`.vscode/launch.json`** - Debug configurations with pre-set environment:
```json
{
    "env": {
        "MPLBACKEND": "Agg",
        "QT_QPA_PLATFORM": "windows",
        "PYTHONUNBUFFERED": "1"
    }
}
```

**`.vscode/settings.json`** - Workspace settings:
```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "terminal.integrated.env.windows": {
        "MPLBACKEND": "Agg"
    }
}
```

**`.env`** - Environment variable file (auto-loaded by VS Code):
```
MPLBACKEND=Agg
QT_QPA_PLATFORM=windows
```

### Layer 3: Runtime Check

At `if __name__ == "__main__"`, we verify and fix environment:
```python
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'Agg'
    # Force matplotlib backend even if already imported
    import matplotlib
    matplotlib.use('Agg', force=True)
```

## How to Use

### Option 1: Use Pre-Configured Debug (Recommended)

1. Open VS Code
2. Open `main.py`
3. Press **F5** or click **Run → Start Debugging**
4. Select: **"Python: Main (Windows Debug Safe)"**

This configuration automatically:
- Sets all environment variables
- Uses correct Python interpreter (venv)
- Configures console output
- Enables full debugging features

### Option 2: Terminal with Debug Attach

1. Open VS Code Terminal
2. Run:
   ```cmd
   venv\Scripts\activate
   python main.py
   ```
3. When running, use **Debug → Attach to Process**

### Option 3: Simple Debug (Minimal Features)

Use configuration: **"Python: Main (Simple - No Debug Features)"**

This has fewer debug features but is most compatible.

## Debug Configurations Explained

### "Python: Main (Windows Debug Safe)"
- **Full debugging** with breakpoints
- All environment variables set
- Uses venv Python
- `justMyCode: false` - can step into libraries
- **Use this** for normal debugging

### "Python: Main (Simple - No Debug Features)"
- **Basic debugging** only
- Minimal environment
- `justMyCode: true` - stays in your code
- **Use this** if full debug mode crashes

### "Python: Current File"
- Debug any Python file
- Minimal configuration
- **Use this** for testing individual modules

## Troubleshooting Debug Mode

### Issue: "No module named 'matplotlib_config'"

**Solution 1:** Ensure `.vscode/settings.json` has correct path:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"
}
```

**Solution 2:** Set working directory in launch.json:
```json
{
    "cwd": "${workspaceFolder}"
}
```

### Issue: Still getting matplotlib Qt backend errors

**Solution:** Add to top of main.py (before imports):
```python
import sys
sys.path.insert(0, os.path.dirname(__file__))
```

### Issue: Breakpoints not working

**Symptom:** Breakpoints are gray/hollow

**Solution:**
1. Check that `justMyCode: false` is set
2. Ensure `.pyc` files are deleted: `find . -name "*.pyc" -delete`
3. Restart VS Code

### Issue: "Debugger is attached" but nothing happens

**Solution:**
1. Stop debugging (Shift+F5)
2. Close all Python processes: `taskkill /F /IM python.exe` (Windows)
3. Restart VS Code
4. Try again

### Issue: Environment variables not loading

**Check:**
```python
# Add at top of main.py to verify
print(f"MPLBACKEND: {os.environ.get('MPLBACKEND')}")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
```

**Solution:** Reload VS Code window:
- Press **Ctrl+Shift+P**
- Type: **"Reload Window"**
- Press Enter

### Issue: "Qt platform plugin 'windows' not found"

**Solution:** Reinstall PySide6 in debug terminal:
```cmd
venv\Scripts\activate
pip install --upgrade --force-reinstall PySide6
```

## VS Code Settings Explained

### `.vscode/launch.json`

Controls how debugger starts:

- `"program"` - Which file to run
- `"console"` - Where output goes (`integratedTerminal` is best)
- `"justMyCode"` - Debug only your code vs all code
- `"env"` - Environment variables
- `"python"` - Which Python interpreter
- `"cwd"` - Working directory

### `.vscode/settings.json`

Workspace-wide settings:

- `"python.defaultInterpreterPath"` - Which Python to use
- `"python.envFile"` - Where to load environment from
- `"terminal.integrated.env.windows"` - Terminal environment
- `"python.analysis.extraPaths"` - Where to find modules

### `.env`

Simple key=value pairs:
```
MPLBACKEND=Agg
PYTHONUNBUFFERED=1
```

Automatically loaded by VS Code Python extension.

## Testing the Fix

### Test 1: Verify Environment in Debug Mode

Add breakpoint at line 10 of main.py, then in Debug Console:
```python
import os
os.environ.get('MPLBACKEND')  # Should return 'Agg'
```

### Test 2: Test matplotlib Import

Add breakpoint after matplotlib_config import:
```python
import matplotlib
matplotlib.get_backend()  # Should return 'Agg'
```

### Test 3: Full Application Debug

1. Set breakpoint in `check_dependencies()` function
2. Press F5
3. Should stop at breakpoint
4. Check Variables panel for all imports
5. Continue (F5) - should complete without errors

## Debug Mode Best Practices

1. **Always use venv Python** - Set in workspace settings
2. **Load .env file** - Place at workspace root
3. **Set environment in launch.json** - Don't rely on .env alone
4. **Use integratedTerminal** - Better than debug console
5. **Clear breakpoints** when not needed - Faster execution

## Advanced: Debug Without Qt GUI

If you want to debug without opening the Qt window:

```python
# In main.py, replace:
launch_qt_viewer()

# With:
print("Debug mode: Skipping Qt window")
# Your debugging code here
```

## Advanced: Profile in Debug Mode

Add to launch.json:
```json
{
    "name": "Python: Profile Main",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/main.py",
    "console": "integratedTerminal",
    "args": [],
    "env": {
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg"
    },
    "profiler": {
        "enabled": true
    }
}
```

## Files Created

- `.vscode/launch.json` - Debug configurations
- `.vscode/settings.json` - Workspace settings
- `.env` - Environment variables
- `DEBUG_MODE_FIX.md` - This file

## Verification Checklist

- [ ] `.vscode/launch.json` exists with correct configurations
- [ ] `.vscode/settings.json` points to venv Python
- [ ] `.env` file has MPLBACKEND=Agg
- [ ] main.py sets environment at module level
- [ ] Can press F5 and debugger starts
- [ ] No matplotlib Qt backend errors
- [ ] Breakpoints work
- [ ] Can step through code

---

**Status:** ✅ FIXED - Multiple layers of protection for VS Code debug mode on Windows


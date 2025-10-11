"""
Matplotlib Configuration for PySide6 Compatibility

CRITICAL: This must be imported BEFORE any matplotlib imports to prevent
the Qt backend conflict with PySide6.

The issue: matplotlib's QtAgg backend looks for QApplication in PySide6.QtGui
but it's actually in PySide6.QtWidgets, causing AttributeError on Windows.

Solution: Force matplotlib to use Agg backend (non-interactive) since we're
using PySide6 for the GUI, not matplotlib's Qt backend.
"""

import os
import sys
import platform


def configure_matplotlib_for_pyside6():
    """
    Configure matplotlib to work with PySide6 without conflicts.
    Must be called before any matplotlib imports.
    """
    # Set matplotlib to use Agg backend (non-interactive)
    # This prevents the Qt backend from conflicting with PySide6
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Also set it programmatically
    import matplotlib
    matplotlib.use('Agg', force=True)
    
    # Suppress matplotlib GUI warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    if platform.system() == 'Windows':
        print("✓ Matplotlib configured for Windows + PySide6 compatibility (Agg backend)")
    
    return True


def fix_matplotlib_qt_backend():
    """
    Alternative: Patch matplotlib to look for QApplication in the correct place.
    This is a more aggressive fix if Agg backend doesn't work.
    """
    try:
        # Try to fix the Qt backend by redirecting imports
        import matplotlib
        matplotlib.use('Agg', force=True)  # Force Agg first
        
        # If user still wants Qt backend, we need to patch it
        # This is commented out by default since Agg works fine
        """
        from PySide6 import QtWidgets, QtGui, QtCore
        
        # Monkey-patch: Add QApplication to QtGui for matplotlib compatibility
        if not hasattr(QtGui, 'QApplication'):
            QtGui.QApplication = QtWidgets.QApplication
        
        if not hasattr(QtGui, 'QMainWindow'):
            QtGui.QMainWindow = QtWidgets.QMainWindow
            
        print("✓ Matplotlib Qt backend patched for PySide6")
        """
        
    except Exception as e:
        print(f"⚠️  Could not patch matplotlib Qt backend: {e}")
        print("   Using Agg backend as fallback")


# Auto-configure on import
try:
    configure_matplotlib_for_pyside6()
except Exception as e:
    print(f"⚠️  Matplotlib configuration warning: {e}")
    print("   Will attempt to use default backend")


if __name__ == "__main__":
    # Test the configuration
    print("Testing matplotlib configuration...")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib backend: {matplotlib.get_backend()}")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib.pyplot imported successfully")
        
        # Test a simple plot (non-interactive)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        print("✓ Plot creation successful")
        
        print("\n" + "="*70)
        print("MATPLOTLIB CONFIGURATION TEST: PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Matplotlib configuration test failed: {e}")
        import traceback
        traceback.print_exc()


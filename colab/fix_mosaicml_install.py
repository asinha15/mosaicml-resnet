#!/usr/bin/env python3
"""
🔧 MosaicML Installation Fix Script for Google Colab
===================================================

This script automatically handles the mosaicml installation issues in Google Colab,
including metadata warnings and version conflicts.

Usage:
    !python fix_mosaicml_install.py
"""

import os
import subprocess
import sys
from typing import List, Tuple

def run_command(cmd: List[str], description: str = "") -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        print(f"🔧 {description}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True, result.stdout
        else:
            print(f"   ⚠️ Warning (exit code {result.returncode})")
            return False, result.stderr
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e)

def test_composer_import() -> bool:
    """Test if composer can be imported successfully."""
    try:
        from composer import Trainer
        from composer.algorithms import MixUp
        print("✅ Composer import test: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Composer import test: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ Composer import test: ERROR - {e}")
        return False

def main():
    """Main installation fix function."""
    print("🚀 MosaicML Composer Installation Fix")
    print("=" * 50)
    
    # Step 1: Upgrade pip to latest version
    print("\n📦 Step 1: Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                "Upgrading pip to latest version")
    
    # Step 2: Test if composer is already working
    print("\n🔍 Step 2: Testing current installation...")
    if test_composer_import():
        print("🎉 Composer is already working! No fix needed.")
        return
    
    # Step 3: Try installing latest mosaicml version
    print("\n📦 Step 3: Installing latest mosaicml version...")
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        "mosaicml>=0.20.0", "--no-warn-conflicts", "--upgrade"
    ], "Installing mosaicml>=0.20.0")
    
    # Test after first attempt
    if test_composer_import():
        print("🎉 Installation successful!")
        return
    
    # Step 4: Try specific version if latest didn't work
    print("\n📦 Step 4: Trying specific working version...")
    run_command([
        sys.executable, "-m", "pip", "uninstall", "mosaicml", "-y"
    ], "Removing previous installation")
    
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        "mosaicml==0.21.0", "--force-reinstall", "--no-deps"
    ], "Installing mosaicml==0.21.0")
    
    # Install dependencies separately
    run_command([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchmetrics", "numpy", "tqdm"
    ], "Installing dependencies")
    
    # Test after second attempt
    if test_composer_import():
        print("🎉 Installation successful with specific version!")
        return
    
    # Step 5: Last resort - install from GitHub source
    print("\n📦 Step 5: Installing from GitHub source (last resort)...")
    run_command([
        sys.executable, "-m", "pip", "uninstall", "mosaicml", "-y"
    ], "Removing previous installation")
    
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/mosaicml/composer.git"
    ], "Installing from GitHub source")
    
    # Final test
    print("\n🔍 Final Test:")
    if test_composer_import():
        print("🎉 SUCCESS: Installation completed from source!")
        print("\n📋 Summary:")
        print("   ✅ MosaicML Composer is now working")
        print("   ✅ Ready to run ImageNet validation training")
    else:
        print("❌ FAILED: Could not install MosaicML Composer")
        print("\n🆘 Manual steps to try:")
        print("   1. Restart Colab runtime: Runtime → Restart runtime")
        print("   2. Try: !pip install --force-reinstall 'mosaicml==0.20.1'")
        print("   3. Check GitHub issues: https://github.com/mosaicml/composer/issues")
        print("   4. Consider using a different Colab runtime")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

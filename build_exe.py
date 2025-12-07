"""
Build script to compile Neural Project into a standalone .exe
"""

import subprocess
import sys
import os

def install_pyinstaller():
    """Install PyInstaller if not available"""
    print("Installing PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

def build_exe():
    """Build the executable"""
    print("\n" + "="*60)
    print("Building Neural Project Executable...")
    print("="*60 + "\n")
    
    # PyInstaller command with options
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window (GUI app)
        "--name=NeuralProject",         # Name of the exe
        "--icon=NONE",                  # No icon (can add later)
        "--add-data", "neural_core;neural_core",  # Include neural_core package
        "--add-data", "adapters;adapters",        # Include adapters package
        "--add-data", "examples;examples",        # Include examples
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=sklearn",
        "--hidden-import=tkinter",
        "--collect-all=numpy",
        "--collect-all=scipy",
        "--collect-all=sklearn",
        "main.py"
    ]
    
    print("Running PyInstaller with command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("✓ BUILD SUCCESSFUL!")
        print("="*60)
        print(f"\nExecutable location: {os.path.join(os.getcwd(), 'dist', 'NeuralProject.exe')}")
        print("\nYou can now run NeuralProject.exe from the 'dist' folder!")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print("✗ BUILD FAILED")
        print("="*60)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Check if PyInstaller is installed
        subprocess.run(["pyinstaller", "--version"], 
                      capture_output=True, check=True)
        print("PyInstaller found!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("PyInstaller not found.")
        install_pyinstaller()
    
    build_exe()

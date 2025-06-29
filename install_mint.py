#!/usr/bin/env python3
"""
Installation script for MINT - Mathematical Intelligence Library.

This script helps users install the mint library and set up the environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header():
    """Print installation header."""
    print("🔧 MINT LIBRARY INSTALLATION")
    print("=" * 60)
    print("Installing MINT - Mathematical Intelligence Library")
    print("For Function Prototype Prompting (FPP)")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("📋 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_mint():
    """Install mint library in development mode."""
    print("\n📦 Installing MINT library...")
    
    try:
        # Install in development mode
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        print("✅ MINT library installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing MINT library: {e}")
        return False


def create_env_file():
    """Create .env file if it doesn't exist."""
    print("\n⚙️ Setting up environment configuration...")
    
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    # Check if env.example exists
    example_path = Path("env.example")
    if not example_path.exists():
        print("❌ env.example file not found")
        return False
    
    # Copy env.example to .env
    try:
        with open(example_path, 'r') as src, open(env_path, 'w') as dst:
            dst.write(src.read())
        
        print("✅ Created .env file from env.example")
        print("📝 Please edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def test_installation():
    """Test if mint library is properly installed."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test import
        import mint
        print("✅ MINT library import successful!")
        
        # Test version
        print(f"   Version: {mint.__version__}")
        
        # Test basic functionality (without API key)
        from mint import get_execution_namespace
        namespace = get_execution_namespace()
        
        if 'add' in namespace and 'sub' in namespace:
            print("✅ Mathematical functions loaded successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Error testing installation: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n\n🎉 MINT INSTALLATION COMPLETED!")
    print("=" * 60)
    print("📚 Next steps:")
    print("   1. Edit .env file and add your OpenAI API key:")
    print("      OPENAI_API_KEY=your_api_key_here")
    print()
    print("   2. Test with simple script:")
    print("      python fpp.py")
    print()
    print("   3. Use in Python code:")
    print("      from mint import solve_math_problem")
    print("      result = solve_math_problem('What is 5 + 3?')")
    print()
    print("   4. Command line interface:")
    print("      mint-fpp solve 'What is 5 + 3?'")
    print("      mint-fpp interactive")
    print()
    print("🔧 Useful commands:")
    print("   • Test installation: python install_mint.py --test-only")
    print("   • Demo: python demo_fpp.py")
    print("   • Full FPP script: python fpp.py")
    print()
    print("📖 Documentation: README.md")


def main():
    """Main installation function."""
    parser = None
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Install MINT library")
        parser.add_argument("--test-only", action="store_true", help="Only test installation")
        args = parser.parse_args()
    except:
        args = None
    
    try:
        print_header()
        
        # Check Python version
        if not check_python_version():
            return False
        
        # Test only mode
        if args and args.test_only:
            return test_installation()
        
        # Install mint library
        if not install_mint():
            return False
        
        # Create .env file
        if not create_env_file():
            print("⚠️ Warning: Could not create .env file")
        
        # Test installation
        if not test_installation():
            print("⚠️ Warning: Installation test failed")
        
        # Print next steps
        print_next_steps()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Installation interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Unexpected error during installation: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
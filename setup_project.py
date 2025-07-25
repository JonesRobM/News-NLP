"""
Project setup script for News Headlines Topic Classifier.

This script helps initialize the project environment, create necessary directories,
and optionally create sample data for testing.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_directories(base_path: str = ".") -> List[str]:
    """
    Create necessary project directories.
    
    Args:
        base_path (str): Base path for the project.
    
    Returns:
        List[str]: List of created directories.
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "outputs",
        "logs",
        "notebooks",
        "src",
        "tests"
    ]
    
    created_dirs = []
    
    for directory in directories:
        dir_path = Path(base_path) / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"📁 Directory already exists: {dir_path}")
    
    return created_dirs


def install_dependencies(requirements_file: str = "requirements.txt") -> bool:
    """
    Install project dependencies from requirements file.
    
    Args:
        requirements_file (str): Path to requirements file.
    
    Returns:
        bool: True if installation successful, False otherwise.
    """
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        print(f"📦 Installing dependencies from {requirements_file}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_sample_data(output_path: str = "data/processed/headlines.csv") -> bool:
    """
    Create sample data for testing.
    
    Args:
        output_path (str): Path to save sample data.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        from utils import create_sample_data
        
        print("🔄 Creating sample data...")
        create_sample_data(output_path, num_samples_per_topic=150)
        print(f"✅ Sample data created: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def run_basic_tests() -> bool:
    """
    Run basic tests to verify installation.
    
    Returns:
        bool: True if tests pass, False otherwise.
    """
    try:
        print("🧪 Running basic tests...")
        
        # Import test module
        import test_basic
        
        # Run tests
        success = test_basic.run_tests()
        
        if success:
            print("✅ All basic tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return success
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def check_dependencies() -> dict:
    """
    Check if required dependencies are available.
    
    Returns:
        dict: Dictionary with dependency status.
    """
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    status = {}
    print("🔍 Checking dependencies...")
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            status[module] = True
            print(f"  ✅ {name}")
        except ImportError:
            status[module] = False
            print(f"  ❌ {name} (not installed)")
    
    return status


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "="*60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. 📊 Collect data using the notebook:")
    print("   jupyter notebook notebooks/data_collection.ipynb")
    print("\n2. 🚀 Train your model:")
    print("   python src/train.py")
    print("\n3. 🔮 Make predictions:")
    print("   python inference.py --interactive")
    print("\n4. 📚 Read the documentation:")
    print("   Check README.md for detailed instructions")
    print("\n5. 🧪 Run tests:")
    print("   python test_basic.py")
    print("\n" + "="*60)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup News Headlines Topic Classifier project"
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--skip-sample-data',
        action='store_true',
        help='Skip sample data creation'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running basic tests'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='.',
        help='Base path for project setup'
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up News Headlines Topic Classifier...")
    print("="*60)
    
    # Step 1: Create directories
    print("\n📁 Creating project directories...")
    created_dirs = create_directories(args.base_path)
    
    # Step 2: Check dependencies
    print(f"\n{'-'*40}")
    dep_status = check_dependencies()
    missing_deps = [name for name, status in dep_status.items() if not status]
    
    # Step 3: Install dependencies if needed
    if missing_deps and not args.skip_deps:
        print(f"\n📦 Installing missing dependencies...")
        success = install_dependencies()
        if not success:
            print("⚠️  Dependency installation failed. You may need to install them manually.")
    elif args.skip_deps:
        print("\n⏭️  Skipping dependency installation (--skip-deps flag)")
    else:
        print("\n✅ All dependencies are already installed!")
    
    # Step 4: Create sample data
    if not args.skip_sample_data:
        print(f"\n{'-'*40}")
        success = create_sample_data()
        if not success:
            print("⚠️  Sample data creation failed. You can create it later manually.")
    else:
        print("\n⏭️  Skipping sample data creation (--skip-sample-data flag)")
    
    # Step 5: Run tests
    if not args.skip_tests:
        print(f"\n{'-'*40}")
        success = run_basic_tests()
        if not success:
            print("⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n⏭️  Skipping tests (--skip-tests flag)")
    
    # Step 6: Display next steps
    display_next_steps()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main data preparation script for DSPy taste classification project.

This script orchestrates the data preparation for both favorite_classifier 
and pairwise_classifier models by calling their respective prepare scripts.
"""

import subprocess
import sys
from pathlib import Path


def run_prepare_script(script_path: Path, classifier_name: str):
    """Run a prepare script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running {classifier_name} prepare script...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings from {classifier_name}:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {classifier_name} prepare script:")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False
    except Exception as e:
        print(f"Unexpected error running {classifier_name} prepare script: {e}")
        return False


def main():
    """Main data preparation orchestration function"""
    print("Starting data preparation for DSPy taste classification project...")
    print("This will run the prepare scripts for both classifiers.")
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / 'scripts'
    
    # Define prepare scripts
    prepare_scripts = [
        (scripts_dir / 'favorite_classifier' / 'prepare.py', 'Favorite Classifier'),
        (scripts_dir / 'pairwise_classifier' / 'prepare.py', 'Pairwise Classifier')
    ]
    
    # Check that all scripts exist
    all_exist = True
    for script_path, name in prepare_scripts:
        if not script_path.exists():
            print(f"ERROR: {name} prepare script not found: {script_path}")
            all_exist = False
    
    if not all_exist:
        print("\nPlease ensure all prepare scripts exist before running this script.")
        sys.exit(1)
    
    # Run each prepare script
    success_count = 0
    for script_path, name in prepare_scripts:
        if run_prepare_script(script_path, name):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully prepared data for {success_count}/{len(prepare_scripts)} classifiers")
    
    if success_count == len(prepare_scripts):
        print("\n✅ All data preparation completed successfully!")
        print("\nNext steps:")
        print("1. Review data splits in data/processed/")
        print("2. Run training script: scripts/favorite_classifier/train.py")
        print("3. Run training script: scripts/pairwise_classifier/train.py")
    else:
        print("\n❌ Some data preparation scripts failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
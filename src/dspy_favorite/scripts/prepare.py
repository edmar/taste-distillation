#!/usr/bin/env python3
"""
DSPy Data preparation script for favorite classifier.

This script is now a thin wrapper around lib/reader/prepare.py for compatibility.
Use lib/reader/prepare.py directly for more control over data preparation.
"""

import os
import sys
from pathlib import Path
import subprocess


def main():
    """Run the unified prepare script for favorite dataset only."""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Path to the unified prepare script
    unified_prepare = project_root / "lib" / "reader" / "prepare.py"
    
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Prepare favorite classification data (wrapper for lib/reader/prepare.py)")
    parser.add_argument('--data', type=str, default='data/raw/reader_export_20250703.csv',
                        help='Path to reader export CSV file')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAVORITE DATASET PREPARATION")
    print("="*60)
    print("Note: This is a compatibility wrapper.")
    print("Consider using lib/reader/prepare.py directly for more options.")
    print("="*60)
    
    # Build command to run unified prepare script
    cmd = [
        sys.executable,
        str(unified_prepare),
        '--data', args.data,
        '--output', args.output,
        '--seed', str(args.seed),
        '--datasets', 'favorite'
    ]
    
    # Run the unified prepare script
    try:
        result = subprocess.run(cmd, check=True)
        
        # Note about output location
        print(f"\n✅ Favorite dataset prepared successfully!")
        print(f"Note: Data is now saved to:")
        print(f"  - {args.output}/reader_favorite/")
        print(f"  - {args.output}/reader_favorite_pairwise/")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error preparing favorite data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
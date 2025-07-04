#!/usr/bin/env python3
"""
Shared logging utilities for DSPy scripts
"""

import os
import sys
from datetime import datetime
from contextlib import contextmanager
from io import StringIO

class TeeOutput:
    """Tee output to both console and log file, filtering repetitive progress lines"""
    
    def __init__(self, log_file_path, show_progress=True):
        self.log_file = open(log_file_path, 'w')
        self.console = sys.stdout
        self.show_progress = show_progress
        
    def write(self, text):
        # Filter progress from console if --no-progress flag is used
        if self.show_progress or self._should_log_line(text):
            self.console.write(text)
        
        # Filter out repetitive progress lines in log file
        if self._should_log_line(text):
            self.log_file.write(text)
            self.log_file.flush()
    
    def _should_log_line(self, text):
        """Determine if a line should be logged to file"""
        # Skip ALL progress lines - don't log any progress bars or metrics to file
        if "Average Metric:" in text:
            return False
        
        # Skip progress bar lines with percentage and speed indicators
        if "|" in text and "%" in text and ("it/s" in text or "s/it" in text):
            return False
            
        # Skip carriage return lines (progress bar updates)
        if text.strip().startswith('\r'):
            return False
        
        # Skip lines that are just whitespace or progress bar characters
        stripped = text.strip()
        if not stripped or stripped in ['\r', '\n', '\r\n']:
            return False
            
        # Log everything else (headers, results, errors, etc.)
        return True
        
    def flush(self):
        self.console.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def get_next_log_path(base_name, log_type):
    """Get the next available log file path with numbering"""
    os.makedirs("saved/logs", exist_ok=True)
    
    counter = 1
    while True:
        log_path = f"saved/logs/{base_name}_{log_type}_{counter:03d}.log"
        if not os.path.exists(log_path):
            return log_path
        counter += 1

@contextmanager
def tee_output(base_name, log_type, show_progress=True):
    """Context manager for tee output to console and log file"""
    log_path = get_next_log_path(base_name, log_type)
    
    # Create log file and add header
    with open(log_path, 'w') as f:
        f.write(f"# {log_type.upper()} LOG\n")
        f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Command: {' '.join(sys.argv)}\n")
        f.write("="*80 + "\n\n")
    
    # Set up tee output
    tee = TeeOutput(log_path, show_progress)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print(f"📝 Logging to: {log_path}")
        yield log_path
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"✅ Log saved to: {log_path}")
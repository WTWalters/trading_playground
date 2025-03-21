#!/usr/bin/env python3
"""
Script to install git hooks for the TITAN Trading System.

This script installs pre-commit hooks to ensure documentation quality and consistency.
It creates a symlink to the hook scripts in the scripts/git_hooks directory.

Usage:
    python install_git_hooks.py
"""

import os
import sys
import stat
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

# Create the pre-commit hook content
PRE_COMMIT_HOOK = """#!/bin/bash
# Pre-commit hook to check documentation standards

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Running documentation checks..."

# Check for broken links in documentation
echo "Checking documentation cross-references..."
python "$SCRIPT_DIR/scripts/check_links.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Documentation contains broken cross-references. Fix before committing."
    exit 1
fi

# Check for naming convention compliance
echo "Checking documentation naming conventions..."
python "$SCRIPT_DIR/scripts/check_doc_naming.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Documentation files don't follow naming conventions. Fix before committing."
    exit 1
fi

# Check for documentation freshness
echo "Checking documentation freshness..."
python "$SCRIPT_DIR/scripts/check_doc_freshness.py"
if [ $? -ne 0 ]; then
    echo "WARNING: Some documentation may be out of date. Consider updating before committing."
    # Don't exit with error for freshness issues, just warn
fi

# All checks passed
echo "All documentation checks passed!"
exit 0
"""

# Create the doc freshness checker script
DOC_FRESHNESS_CHECKER = """#!/usr/bin/env python3
\"\"\"
Script to check if documentation is up-to-date with code changes.

This script compares the modification dates of source code files and their
corresponding documentation files to identify potentially outdated documentation.

Usage:
    python check_doc_freshness.py [threshold_days]
    
The threshold_days parameter specifies how many days old a documentation file can be
relative to its code before being considered outdated. Default is 30 days.
\"\"\"

import os
import sys
import argparse
import datetime
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

# Define the mapping of code directories to documentation files
CODE_TO_DOC_MAPPING = {
    "src/data/ingestion": "docs/components/data/data_ingestion.md",
    "src/analysis/cointegration": "docs/components/analysis/cointegration_analysis.md",
    "src/analysis/parameters": "docs/components/analysis/parameter_management.md",
    "src/analysis/regime": "docs/components/analysis/regime_detection.md",
    "src/backtesting/engine": "docs/components/backtesting/backtesting_framework.md",
    "src/backtesting/walk_forward": "docs/components/backtesting/walk_forward_testing.md",
    "src/trading/risk": "docs/components/trading/risk_controls.md",
    "src/trading/position": "docs/components/trading/position_sizing.md",
}

def get_latest_modification_time(directory):
    \"\"\"Get the latest modification time of any file in the directory.\"\"\"
    latest_mtime = 0
    
    # Skip if directory doesn't exist
    if not os.path.exists(directory):
        return latest_mtime
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                mtime = os.path.getmtime(filepath)
                if mtime > latest_mtime:
                    latest_mtime = mtime
    
    return latest_mtime

def check_doc_freshness(threshold_days=30):
    \"\"\"
    Check if documentation is up-to-date with code changes.
    
    Args:
        threshold_days: Maximum age of documentation relative to code (in days)
        
    Returns:
        bool: True if all documentation is up-to-date, False otherwise
    \"\"\"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    outdated_docs = []
    
    for code_dir, doc_file in CODE_TO_DOC_MAPPING.items():
        code_path = os.path.join(project_root, code_dir)
        doc_path = os.path.join(project_root, doc_file)
        
        # Skip if code directory or doc file doesn't exist
        if not os.path.exists(code_path) or not os.path.exists(doc_path):
            continue
        
        # Get latest modification time of code
        code_mtime = get_latest_modification_time(code_path)
        if code_mtime == 0:
            continue
            
        # Get modification time of documentation
        doc_mtime = os.path.getmtime(doc_path)
        
        # Convert to datetime objects
        code_time = datetime.datetime.fromtimestamp(code_mtime)
        doc_time = datetime.datetime.fromtimestamp(doc_mtime)
        
        # Calculate age
        age = code_time - doc_time
        
        # Check if documentation is older than code by more than threshold
        if age.days > threshold_days:
            outdated_docs.append((doc_file, age.days))
    
    # Report outdated documentation
    if outdated_docs:
        print(f"{Fore.YELLOW}Found {len(outdated_docs)} outdated documentation files:{Style.RESET_ALL}")
        for doc, age in outdated_docs:
            print(f"{Fore.YELLOW}- {doc}: {age} days older than corresponding code{Style.RESET_ALL}")
        return False
    else:
        print(f"{Fore.GREEN}All documentation is up-to-date with code changes.{Style.RESET_ALL}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Check if documentation is up-to-date with code changes.')
    parser.add_argument('threshold_days', nargs='?', type=int, default=30, 
                        help='Maximum age of documentation relative to code (in days)')
    args = parser.parse_args()
    
    success = check_doc_freshness(args.threshold_days)
    
    if not success:
        print(f"{Fore.YELLOW}Warning: Some documentation may be out of date.{Style.RESET_ALL}")
        return 1
    else:
        return 0

if __name__ == '__main__':
    sys.exit(main())
"""

def install_git_hooks():
    """Install Git hooks to the project's .git/hooks directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    git_hooks_dir = os.path.join(project_root, ".git", "hooks")
    
    # Ensure git hooks directory exists
    if not os.path.exists(git_hooks_dir):
        print(f"{Fore.RED}Error: Git hooks directory not found at {git_hooks_dir}.{Style.RESET_ALL}")
        print(f"{Fore.RED}Are you sure this is a Git repository?{Style.RESET_ALL}")
        return False
    
    # Create the pre-commit hook file
    pre_commit_path = os.path.join(git_hooks_dir, "pre-commit")
    try:
        with open(pre_commit_path, 'w') as f:
            f.write(PRE_COMMIT_HOOK)
        
        # Make it executable
        os.chmod(pre_commit_path, os.stat(pre_commit_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"{Fore.GREEN}Successfully installed pre-commit hook to {pre_commit_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error installing pre-commit hook: {e}{Style.RESET_ALL}")
        return False
    
    # Create the documentation freshness checker script
    freshness_checker_path = os.path.join(script_dir, "check_doc_freshness.py")
    try:
        with open(freshness_checker_path, 'w') as f:
            f.write(DOC_FRESHNESS_CHECKER)
        
        # Make it executable
        os.chmod(freshness_checker_path, os.stat(freshness_checker_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"{Fore.GREEN}Successfully created documentation freshness checker at {freshness_checker_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error creating documentation freshness checker: {e}{Style.RESET_ALL}")
        return False
    
    print(f"\n{Fore.GREEN}Git hooks installation completed successfully.{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Now, documentation will be checked for:"){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}1. Broken cross-references{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}2. Naming convention compliance{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}3. Freshness relative to code changes{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}These checks will run automatically before each commit.{Style.RESET_ALL}")
    
    return True

if __name__ == "__main__":
    success = install_git_hooks()
    sys.exit(0 if success else 1)

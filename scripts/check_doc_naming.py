#!/usr/bin/env python3
"""
Script to check documentation file naming conventions.

This script verifies that all markdown files in the documentation directory follow
the naming conventions defined in the TITAN Trading System documentation standards.

Usage:
    python check_doc_naming.py [directory_path]
    
If no directory is provided, defaults to the docs directory.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

def check_naming_conventions(directory):
    """
    Check if all markdown files follow the naming conventions.
    
    Naming conventions:
    1. All files use snake_case (underscores between words)
    2. All files have a .md extension
    3. No uppercase letters in filenames
    4. No spaces in filenames
    5. No special characters except underscores
    """
    violations = []
    
    # Regular expression for valid snake_case filenames
    valid_name_pattern = re.compile(r'^[a-z0-9_]+\.md$')
    
    # Exceptions to the naming conventions
    exceptions = {'README.md', 'LICENSE.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md'}
    
    # Check all files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.md'):
                continue
                
            # Skip exception files
            if filename in exceptions:
                continue
                
            # Check if filename follows conventions
            if not valid_name_pattern.match(filename):
                # Identify specific violations
                violations_found = []
                
                if ' ' in filename:
                    violations_found.append("contains spaces")
                    
                if re.search(r'[A-Z]', filename):
                    violations_found.append("contains uppercase letters")
                    
                if re.search(r'[^a-zA-Z0-9_\.]', filename):
                    violations_found.append("contains special characters")
                    
                if '-' in filename:
                    violations_found.append("uses hyphens instead of underscores")
                    
                if not filename.endswith('.md'):
                    violations_found.append("doesn't have .md extension")
                
                violations.append((os.path.join(root, filename), ', '.join(violations_found)))
    
    # Report violations
    if violations:
        print(f"{Fore.RED}Found {len(violations)} files that don't follow naming conventions:{Style.RESET_ALL}")
        for filepath, reason in violations:
            rel_path = os.path.relpath(filepath, directory)
            print(f"{Fore.RED}- {rel_path}: {reason}{Style.RESET_ALL}")
            
        # Suggest corrections
        print(f"\n{Fore.YELLOW}Suggested corrections:{Style.RESET_ALL}")
        for filepath, _ in violations:
            filename = os.path.basename(filepath)
            rel_path = os.path.relpath(filepath, directory)
            
            # Generate corrected filename
            corrected = filename.lower()
            corrected = re.sub(r'[^a-z0-9_\.]', '_', corrected)
            corrected = re.sub(r'-', '_', corrected)
            corrected = re.sub(r'\..*$', '.md', corrected)
            corrected = re.sub(r'_+', '_', corrected)
            
            print(f"{Fore.YELLOW}- {rel_path} → {os.path.join(os.path.dirname(rel_path), corrected)}{Style.RESET_ALL}")
        
        return False
    else:
        print(f"{Fore.GREEN}All documentation files follow naming conventions.{Style.RESET_ALL}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Check documentation file naming conventions.')
    parser.add_argument('directory', nargs='?', default=None, help='Directory to check (default: docs)')
    args = parser.parse_args()
    
    # Determine directory to check
    if args.directory:
        directory = args.directory
    else:
        # Default to the docs directory in the project
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        directory = os.path.join(project_root, 'docs')
    
    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"{Fore.RED}Error: Directory {directory} does not exist.{Style.RESET_ALL}")
        return 1
    
    success = check_naming_conventions(directory)
    
    if not success:
        print(f"\n{Fore.RED}Naming convention check failed. Please fix the issues before committing.{Style.RESET_ALL}")
        return 1
    else:
        return 0

if __name__ == '__main__':
    sys.exit(main())

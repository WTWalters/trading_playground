#!/usr/bin/env python3
"""
Script to verify cross-references in documentation.

This script checks all markdown files in the specified directory (or docs by default)
and verifies that all relative links point to existing files. It handles both
direct file links and anchor links within files.

Usage:
    python check_links.py [directory_path]
    
If no directory is provided, defaults to the docs directory.
"""

import os
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

def find_markdown_files(directory):
    """Find all markdown files in the directory and subdirectories."""
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def extract_links(filepath):
    """Extract all markdown links from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to find markdown links
    # Captures: [link text](link url)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    links = []
    for match in link_pattern.finditer(content):
        link_text, link_url = match.groups()
        links.append((link_text, link_url, match.start()))
    
    return links

def get_anchors(filepath):
    """Extract all heading anchors from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to find markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
    
    anchors = []
    for match in heading_pattern.finditer(content):
        level, heading_text = match.groups()
        # Generate GitHub-style anchor
        anchor = heading_text.lower()
        # Replace non-alphanumeric characters with -
        anchor = re.sub(r'[^\w\-\s]', '', anchor)
        # Replace spaces with -
        anchor = re.sub(r'\s+', '-', anchor)
        # Remove consecutive -
        anchor = re.sub(r'-+', '-', anchor)
        # Remove leading and trailing -
        anchor = anchor.strip('-')
        
        anchors.append(f"#{anchor}")
    
    return anchors

def check_links(directory):
    """Check all markdown links in a directory."""
    print(f"{Fore.BLUE}Scanning for markdown files in {directory}...{Style.RESET_ALL}")
    
    # Find all markdown files
    markdown_files = find_markdown_files(directory)
    print(f"{Fore.GREEN}Found {len(markdown_files)} markdown files.{Style.RESET_ALL}")
    
    # Map of file paths to their anchors
    file_anchors = {}
    
    # Extract all anchors from markdown files
    for filepath in markdown_files:
        anchors = get_anchors(filepath)
        file_anchors[filepath] = anchors
    
    broken_links = []
    warning_links = []
    valid_links = 0
    file_links = defaultdict(list)
    
    # Check all links in each file
    for filepath in markdown_files:
        links = extract_links(filepath)
        
        for link_text, link_url, position in links:
            # Skip external links
            if link_url.startswith(('http://', 'https://')):
                continue
            
            # Handle anchor-only links
            if link_url.startswith('#'):
                anchor = link_url
                if anchor in file_anchors[filepath]:
                    valid_links += 1
                else:
                    broken_links.append((filepath, link_text, link_url, position, "Anchor not found in file"))
                continue
            
            # Split URL and anchor
            url_parts = link_url.split('#', 1)
            raw_url = url_parts[0]
            anchor = '#' + url_parts[1] if len(url_parts) > 1 else None
            
            # Resolve relative path
            if not raw_url.startswith('/'):
                target_path = os.path.normpath(os.path.join(os.path.dirname(filepath), raw_url))
            else:
                # Remove leading slash for paths starting with /
                target_path = os.path.normpath(os.path.join(directory, raw_url.lstrip('/')))
            
            # Check if file exists
            if not os.path.exists(target_path):
                broken_links.append((filepath, link_text, link_url, position, "Target file not found"))
                continue
            
            # If there's an anchor, check if it exists in the target file
            if anchor and target_path in file_anchors:
                if anchor not in file_anchors[target_path]:
                    broken_links.append((filepath, link_text, link_url, position, "Anchor not found in target file"))
                    continue
            
            # If we got here, the link is valid
            valid_links += 1
            file_links[filepath].append((link_text, link_url))
            
            # Issue a warning for .md files that don't use relative paths
            if raw_url.endswith('.md') and '../' not in raw_url and not raw_url.startswith('/'):
                warning_links.append((filepath, link_text, link_url, position, "Consider using relative path with ../"))
    
    # Output results
    if broken_links:
        print(f"\n{Fore.RED}Found {len(broken_links)} broken links:{Style.RESET_ALL}")
        for source, text, url, position, reason in broken_links:
            rel_source = os.path.relpath(source, directory)
            print(f"{Fore.RED}In {rel_source}: Link '{text}' to '{url}' - {reason}{Style.RESET_ALL}")
    
    if warning_links:
        print(f"\n{Fore.YELLOW}Found {len(warning_links)} links with warnings:{Style.RESET_ALL}")
        for source, text, url, position, reason in warning_links:
            rel_source = os.path.relpath(source, directory)
            print(f"{Fore.YELLOW}In {rel_source}: Link '{text}' to '{url}' - {reason}{Style.RESET_ALL}")
    
    # Report files with no outgoing links
    files_without_links = [f for f in markdown_files if f not in file_links]
    if files_without_links:
        print(f"\n{Fore.YELLOW}Found {len(files_without_links)} files with no outgoing links:{Style.RESET_ALL}")
        for filepath in files_without_links:
            rel_path = os.path.relpath(filepath, directory)
            print(f"{Fore.YELLOW}- {rel_path}{Style.RESET_ALL}")
    
    # Print summary
    print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}")
    print(f"- Checked {len(markdown_files)} markdown files")
    print(f"- Found {valid_links} valid links")
    print(f"- Found {len(broken_links)} broken links")
    print(f"- Found {len(warning_links)} links with warnings")
    print(f"- Found {len(files_without_links)} files with no outgoing links")
    
    return len(broken_links) == 0

def main():
    parser = argparse.ArgumentParser(description='Check markdown links in documentation.')
    parser.add_argument('directory', nargs='?', default=None, help='Directory to check (default: docs)')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix broken links (experimental)')
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
    
    success = check_links(directory)
    
    if not success:
        print(f"\n{Fore.RED}Link check failed. Please fix broken links before committing.{Style.RESET_ALL}")
        return 1
    else:
        print(f"\n{Fore.GREEN}All links are valid!{Style.RESET_ALL}")
        return 0

if __name__ == '__main__':
    sys.exit(main())

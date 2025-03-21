#!/usr/bin/env python3
"""
Script to run the regime transition tests and generate a comprehensive report.

This script executes the test_regime_transition.py tests and collects the results
including all visualizations for analysis.
"""

import os
import unittest
import sys
import datetime
import argparse
import logging
from pathlib import Path

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the test module
from test_regime_transition import TestRegimeTransition

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('regime_tests.log')
        ]
    )
    return logging.getLogger(__name__)

def create_report_directory():
    """Create a directory for the test report."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"test_reports/regime_tests_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir

def run_specific_test(test_class, test_name, logger):
    """Run a specific test method."""
    suite = unittest.TestSuite()
    suite.addTest(test_class(test_name))

    runner = unittest.TextTestRunner(verbosity=2)
    logger.info(f"Running test: {test_name}")
    result = runner.run(suite)

    return result.wasSuccessful()

def run_all_tests(logger):
    """Run all tests in the TestRegimeTransition class."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegimeTransition)

    runner = unittest.TextTestRunner(verbosity=2)
    logger.info("Running all regime transition tests")
    result = runner.run(suite)

    return result.wasSuccessful()

def copy_visualizations(test_output_dir, report_dir, logger):
    """Copy test visualizations to the report directory."""
    import shutil

    try:
        # Get all PNG files from the test output directory
        output_dir = Path(test_output_dir)
        png_files = list(output_dir.glob("*.png"))

        logger.info(f"Found {len(png_files)} visualization files")

        # Copy each file to the report directory
        for png_file in png_files:
            shutil.copy2(png_file, report_dir)
            logger.info(f"Copied {png_file.name} to report directory")

        return True
    except Exception as e:
        logger.error(f"Error copying visualizations: {str(e)}")
        return False

def generate_html_report(report_dir, success, logger):
    """Generate an HTML report with test results and visualizations."""
    html_path = report_dir / "report.html"

    # Get all PNG files in the report directory
    png_files = list(report_dir.glob("*.png"))
    png_files.sort()

    # Create a simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Regime Transition Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            .visualization {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
            .visualization img {{ max-width: 100%; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .timestamp {{ color: #7f8c8d; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>Adaptive Parameter Management: Regime Transition Test Report</h1>
        <p class="timestamp">Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Test Results</h2>
        <p>Overall Status: <span class="{'pass' if success else 'fail'}">{
            'PASSED' if success else 'FAILED'}</span></p>

        <h2>Visualizations</h2>
    """

    # Add visualizations to the report
    for png_file in png_files:
        html_content += f"""
        <div class="visualization">
            <h3>{png_file.stem.replace('_', ' ').title()}</h3>
            <img src="{png_file.name}" alt="{png_file.stem}">
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Write the HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Generated HTML report at {html_path}")
    return html_path

def main():
    """Main function to run the tests and generate the report."""
    parser = argparse.ArgumentParser(description='Run regime transition tests')
    parser.add_argument('--test', help='Run a specific test (leave empty to run all tests)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)

    logger.info("Starting regime transition test run")

    # Create report directory
    report_dir = create_report_directory()
    logger.info(f"Created report directory: {report_dir}")

    # Initialize test instance to get output directory
    test_instance = TestRegimeTransition()
    test_output_dir = test_instance.output_dir

    # Run tests
    if args.test:
        # Run specific test
        success = run_specific_test(TestRegimeTransition, args.test, logger)
    else:
        # Run all tests
        success = run_all_tests(logger)

    # Copy visualizations to report directory
    copy_visualizations(test_output_dir, report_dir, logger)

    # Generate HTML report
    html_path = generate_html_report(report_dir, success, logger)

    logger.info(f"Test run completed with status: {'SUCCESS' if success else 'FAILURE'}")
    logger.info(f"Report available at: {html_path}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

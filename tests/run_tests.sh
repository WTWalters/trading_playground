#!/bin/bash
# Script to run the TITAN regime transition tests and open the generated report

set -e  # Exit on error

# Display header
echo "====================================================="
echo "TITAN Trading System - Regime Transition Test Runner"
echo "====================================================="

# Create a directory for the tests if it doesn't exist
mkdir -p tests/test_outputs

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Parse arguments
DEBUG_FLAG=""
TEST_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --test)
      TEST_NAME="--test $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--debug] [--test TEST_NAME]"
      exit 1
      ;;
  esac
done

# Run the tests
echo "Running regime transition tests..."
python "$SCRIPT_DIR/run_regime_tests.py" $DEBUG_FLAG $TEST_NAME

# Get the latest report directory
LATEST_REPORT=$(find "$SCRIPT_DIR/test_reports" -type d -name "regime_tests_*" | sort -r | head -n 1)

if [ -z "$LATEST_REPORT" ]; then
    echo "Error: No test report was generated."
    exit 1
fi

HTML_REPORT="$LATEST_REPORT/report.html"

if [ ! -f "$HTML_REPORT" ]; then
    echo "Error: HTML report not found at $HTML_REPORT"
    exit 1
fi

echo ""
echo "Test execution completed."
echo "Report generated at: $HTML_REPORT"
echo ""

# Try to open the HTML report in a browser
if command -v xdg-open &> /dev/null; then
    echo "Opening report in browser..."
    xdg-open "$HTML_REPORT"
elif command -v open &> /dev/null; then
    echo "Opening report in browser..."
    open "$HTML_REPORT"
else
    echo "Could not automatically open the report."
    echo "Please open the following file in your browser:"
    echo "$HTML_REPORT"
fi

echo ""
echo "Done!"

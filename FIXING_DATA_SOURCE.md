# Fixing the Data Source Issue in TITAN Trading Platform

## Problem Diagnosis

We've identified that the backtesting component is failing to find data despite the cointegration analysis working correctly. The root cause is a source parameter mismatch - the synthetic data is loaded with a 'synthetic' source identifier, but the backtest wasn't explicitly specifying this source when querying data.

## How to Fix and Run the System

Follow these steps to run the system with the fixes applied:

### Step 1: Run the Debug Script to Confirm Data Availability

```bash
python scripts/run_debug_backtest.py
```

This will check data availability for all symbols with the 'synthetic' source and print date ranges, helping confirm that data exists.

### Step 2: Run the Cointegration Analysis 

```bash
python scripts/run_complete_test.py
```

This has been fixed to explicitly use the 'synthetic' source and should work correctly.

### Step 3: Run the Fixed Backtest

```bash
python scripts/run_backtest.py
```

We've updated this script to:
1. Dynamically detect the date range where synthetic data exists
2. Explicitly pass the 'synthetic' source parameter 
3. Handle column naming issues in the data frames

### Step 4: Run the Robust Pipeline (Optional)

For a more comprehensive run with better error handling:

```bash
python scripts/run_robust_pipeline.py
```

This improved pipeline:
1. Continues execution even if individual components fail
2. Provides detailed logging
3. Generates a comprehensive report regardless of errors

## Technical Details of the Fix

1. **Root Cause**: Source parameter mismatch between data loading and data retrieval
2. **Key Changes**:
   - Added explicit 'synthetic' source parameter in all database queries
   - Improved date range detection to match the actual data
   - Enhanced error handling and logging
   - Fixed column name handling for correlation metrics

## Testing the Fixes

If you want to validate that the fixes are working:

1. Check that the cointegration analysis finds pairs (should find 7 cointegrated pairs)
2. Verify that the backtest can now retrieve price data (no more "0 rows" messages)
3. Confirm that the full pipeline runs end-to-end without errors

## Debugging Notes

If you still encounter issues:
- Look for "source" related messages in the logs
- Check the date ranges being used for queries
- Verify that the synthetic data was loaded properly

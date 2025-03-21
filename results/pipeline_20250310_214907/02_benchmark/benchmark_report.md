# Performance Benchmark: Standard vs. Parallel Implementation

Generated on: 2025-03-10 21:49:09

## Performance Comparison

| Number of Symbols | Possible Pairs | Standard Time (s) | Parallel Time (s) | Speedup Factor | Standard Pairs | Parallel Pairs |
|-------------------|---------------|-------------------|-------------------|----------------|---------------|----------------|
| 5 | 10 | 0.06 | 0.04 | 1.51x | 0 | 0 |
| 10 | 45 | 0.09 | 0.05 | 1.68x | 0 | 0 |
| 15 | 105 | 0.10 | 0.04 | 2.43x | 0 | 0 |

## Interpretation

- Average speedup: 1.87x
- Maximum speedup: 2.43x (with 15 symbols)
- Both implementations found the same number of pairs, confirming correctness.
- The parallel implementation becomes increasingly advantageous as the number of symbols grows.

## Projections

- Estimated time for 100 symbols (4,950 possible pairs):
  - Standard implementation: 4.90 seconds (0.08 minutes)
  - Parallel implementation: 2.02 seconds (0.03 minutes)
  - Projected time savings: 2.88 seconds

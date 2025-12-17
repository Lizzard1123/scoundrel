# MCTS Performance Optimization TODO

This directory contains detailed analysis of performance bottlenecks identified in the MCTS implementation for Scoundrel.

## Overview

The MCTS agent currently runs 10,000 simulations per move, which can be slow. This directory documents 12 identified optimization opportunities, prioritized by estimated impact.

## Priority Ranking

### CRITICAL (50-80% speedup potential)
1. **[Excessive Deep Copying](01-excessive-deep-copying.md)** - 20,000 deep copies per move
2. **[Unnecessary GameManager Initialization](02-unnecessary-gamemanager-initialization.md)** - Full initialization for MCTS
3. **[Repeated Engine Creation](03-repeated-engine-creation.md)** - New engine for every action

### HIGH (10-30% speedup potential)
4. **[Inefficient State Hashing](04-inefficient-state-hashing.md)** - String concatenation overhead
5. **[Translator Overhead](05-translator-overhead.md)** - Unnecessary tensor creation
10. **[Determinization Inefficiency](10-determinization-inefficiency.md)** - Always deep copies

### MEDIUM (5-15% speedup potential)
6. **[Heuristic Policy Overhead](06-heuristic-policy-overhead.md)** - Complex calculations
7. **[Parallelization Overhead](07-parallelization-overhead.md)** - Process overhead
8. **[No State Reuse/Caching](08-no-state-reuse-caching.md)** - Redundant computations
9. **[Score Calculation Overhead](09-score-calculation-overhead.md)** - Property recalculation
11. **[Simulation Depth Limit](11-simulation-depth-limit.md)** - Excessive max depth
12. **[No Early Termination](12-no-early-termination.md)** - Wasted computation

## Quick Start

1. Start with **Issue #1** (Deep Copying) - highest impact
2. Then tackle **Issue #2** (GameManager) and **Issue #3** (Engine Creation)
3. Work through HIGH priority items
4. Finish with MEDIUM priority optimizations

## Expected Overall Impact

If all optimizations are implemented:
- **Estimated total speedup: 3-10x faster**
- Critical issues alone: **2-5x faster**
- Combined with high/medium: **5-10x faster**

## Notes

- Each issue file contains:
  - Problem description
  - Root cause analysis
  - Current code locations
  - Impact analysis
  - Proposed solutions
  - Recommended approach
  - Implementation notes
  - Testing considerations

- Test after each optimization to verify:
  - Correctness (same game outcomes)
  - Performance improvement
  - No regressions

## Current Configuration

- Simulations per move: 10,000 (`MCTS_NUM_SIMULATIONS`)
- Max depth: 200 (`MCTS_MAX_DEPTH`)
- Workers: 8 (`MCTS_NUM_WORKERS`)
- Random rollout: True (`USE_RANDOM_ROLLOUT`)

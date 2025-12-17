# Issue #7: Parallelization Overhead

## Priority: MEDIUM
**Estimated Impact: Variable (may help or hurt depending on workload)**

## Problem Description

Using `ProcessPoolExecutor` with 'spawn' context has overhead:

- Each worker process must pickle/unpickle GameState
- Pickling involves serialization overhead
- Each worker creates new MCTSAgent instance
- Communication overhead between processes
- Comment mentions "inflection point around 2000 simulations"

## Root Cause

- Process-based parallelism requires serialization
- 'spawn' context (required on macOS) is slower than 'fork'
- Overhead may outweigh benefits for smaller workloads

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:163-203` - `_parallel_search()` method
- `scoundrel/rl/mcts/mcts_agent.py:25-66` - `_run_worker_simulations()` worker function
- `scoundrel/rl/mcts/constants.py:22` - `MCTS_NUM_WORKERS = 8`

## Impact Analysis

- **Frequency**: Once per move (when parallelization enabled)
- **Cost**: 
  - Process creation overhead
  - State serialization (pickling)
  - Inter-process communication
  - Result aggregation
- **Total overhead**: May be negative for <2000 simulations

## Proposed Solutions

### Option 1: Threading Instead of Multiprocessing
- Use `ThreadPoolExecutor` instead of `ProcessPoolExecutor`
- No serialization overhead
- Shared memory (but need GIL consideration)
- May work if operations release GIL

### Option 2: Adaptive Parallelization
- Only use parallelization above threshold (e.g., 2000 simulations)
- Use sequential for smaller workloads
- Dynamic worker count based on simulation count

### Option 3: Optimize Serialization
- Use faster serialization (pickle protocol 5, cloudpickle)
- Minimize data sent to workers
- Send only necessary state information

### Option 4: Shared Memory Multiprocessing
- Use shared memory for GameState
- Reduce serialization overhead
- More complex but potentially faster

### Option 5: Keep Current (May Be Optimal)
- Current implementation may already be optimal
- Test different worker counts
- Measure actual speedup

## Recommended Approach

**Option 2** (Adaptive Parallelization) - Best balance:
- Use parallelization when it helps
- Avoid overhead when it hurts
- Simple to implement

## Implementation Notes

- Add threshold check before parallelization
- Consider CPU count and simulation count
- Measure overhead vs benefit

## Testing Considerations

- Benchmark sequential vs parallel for different simulation counts
- Test with different worker counts (1, 4, 8, 16)
- Measure actual speedup
- Verify results are identical

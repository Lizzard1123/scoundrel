# Issue #5: Parallelization Optimization (MEDIUM)

## Priority: MEDIUM
**Estimated Impact: Variable (may help or hurt depending on workload)**

## Problem Description

Using `ProcessPoolExecutor` with 'spawn' context has overhead:

- Each worker process must pickle/unpickle GameState
- Pickling involves serialization overhead
- Each worker creates new MCTSAgent instance
- Communication overhead between processes
- Comment mentions "inflection point around 2000 simulations" but current config is 80,000

## Root Cause

- Process-based parallelism requires serialization
- 'spawn' context (required on macOS) is slower than 'fork'
- Overhead may outweigh benefits for smaller workloads
- With 80,000 simulations, parallelization likely helps, but overhead still exists

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:269-315` - `_parallel_search()` method
- `scoundrel/rl/mcts/mcts_agent.py:113-158` - `_run_worker_simulations()` worker function
- `scoundrel/rl/mcts/constants.py:22` - `MCTS_NUM_WORKERS = 8`

## Impact Analysis

- **Frequency**: Once per move (when parallelization enabled)
- **Cost**: 
  - Process creation overhead
  - State serialization (pickling GameState)
  - Inter-process communication
  - Result aggregation
- **Total overhead**: May be negative for <2000 simulations, but likely positive for 80,000

## Proposed Solutions

### Option 1: Adaptive Parallelization (RECOMMENDED)
- Only use parallelization above threshold (e.g., 2000 simulations)
- Use sequential for smaller workloads
- Dynamic worker count based on simulation count
- Simple to implement

### Option 2: Optimize Serialization
- Use faster serialization (pickle protocol 5, cloudpickle)
- Minimize data sent to workers
- Send only necessary state information
- Reduce pickling overhead

### Option 3: Shared Memory Multiprocessing
- Use shared memory for GameState
- Reduce serialization overhead
- More complex but potentially faster

### Option 4: Threading Instead of Multiprocessing
- Use `ThreadPoolExecutor` instead of `ProcessPoolExecutor`
- No serialization overhead
- Shared memory (but need GIL consideration)
- May work if operations release GIL

### Option 5: Keep Current (May Be Optimal)
- Current implementation may already be optimal for 80,000 simulations
- Test different worker counts
- Measure actual speedup

## Recommended Approach

**Option 1** (Adaptive Parallelization) - Best balance:
- Use parallelization when it helps
- Avoid overhead when it hurts
- Simple to implement
- Can combine with Option 2 for better results

## Implementation Notes

1. Add threshold check before parallelization:
   ```python
   if self.num_simulations < 2000 or self.num_workers <= 1:
       return self._sequential_search(game_state)
   ```

2. Consider CPU count and simulation count
3. Measure overhead vs benefit
4. Optionally optimize serialization if needed

## Testing Considerations

- Benchmark sequential vs parallel for different simulation counts
- Test with different worker counts (1, 4, 8, 16)
- Measure actual speedup
- Verify results are identical
- Test on different platforms (macOS, Linux)

## Expected Outcome

- **Performance**: Better scaling for different workloads
- **Code**: Slightly more complex (threshold logic)
- **Risk**: Low-Medium (need to test thoroughly)

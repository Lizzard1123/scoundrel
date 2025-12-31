"""
Pytest test for comparing inference performance between policy small and large models.

This test measures inference time differences between small and large policy models
by testing both networks on the same random game states from MCTS logs.

To run this test with full output (including skip reasons and print statements):
    pytest scoundrel/rl/alpha_scoundrel/policy/tests/test_inference_performance.py -v -s -rs

Note: Architecture constants are automatically inferred from checkpoint state dicts
by the inference classes. No manual configuration needed - the test adapts automatically
to architecture changes.
"""
import json
import random
import time
import statistics
from pathlib import Path
from typing import List

import pytest

from scoundrel.models.game_state import GameState
from scoundrel.rl.alpha_scoundrel.policy.policy_small.inference import PolicySmallInference
from scoundrel.rl.alpha_scoundrel.policy.policy_large.inference import PolicyLargeInference
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import DEFAULT_MCTS_LOGS_DIR
from scoundrel.rl.alpha_scoundrel.data_utils import deserialize_game_state


def load_random_game_states(
    log_dir: Path,
    num_states: int = 20,
    seed: int = 42
) -> List[GameState]:
    """
    Load random game states from MCTS log files.
    
    Args:
        log_dir: Directory containing MCTS log JSON files
        num_states: Number of random game states to load
        seed: Random seed for reproducibility
        
    Returns:
        List of GameState objects
    """
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("*.json"))
    
    if not log_files:
        raise ValueError(f"No JSON files found in {log_dir}")
    
    random.seed(seed)
    game_states = []
    
    # Collect all game states from all log files
    all_states = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                game_data = json.load(f)
            
            events = game_data.get("events", [])
            for event in events:
                try:
                    game_state_dict = event.get("game_state", {})
                    if game_state_dict:
                        game_state = deserialize_game_state(game_state_dict)
                        all_states.append(game_state)
                except Exception as e:
                    continue
        except Exception as e:
            continue
    
    if len(all_states) < num_states:
        raise ValueError(
            f"Only found {len(all_states)} game states, but need {num_states}. "
            f"Please ensure there are enough game states in {log_dir}"
        )
    
    # Randomly sample the requested number of states
    game_states = random.sample(all_states, num_states)
    
    return game_states


@pytest.mark.parametrize("small_model_path,large_model_path", [
    (
        "scoundrel/rl/alpha_scoundrel/policy/policy_small/checkpoints/run_20251229_184938/policy_small_epoch_10.pt",
        "scoundrel/rl/alpha_scoundrel/policy/policy_large/checkpoints/run_20251230_215704/policy_large_epoch_400.pt"
    ),
])
def test_inference_time_comparison(small_model_path: str, large_model_path: str):
    """
    Test inference time difference between small and large policy models.
    
    Measures inference time for both models on the same random game states
    from MCTS logs, averaging across 20 game states for consistency.
    
    Architecture constants are automatically inferred from checkpoint state dicts
    by the inference classes, so no manual configuration is needed when architectures change.
    
    Args:
        small_model_path: Path to small model checkpoint (relative to workspace root)
        large_model_path: Path to large model checkpoint (relative to workspace root)
    """
    # Resolve paths relative to workspace root
    # Go up from tests -> policy -> alpha_scoundrel -> rl -> scoundrel (package) -> scoundrel (workspace root)
    test_file_dir = Path(__file__).parent.resolve()
    workspace_root = test_file_dir.parent.parent.parent.parent.parent.resolve()
    
    # Convert to Path objects and resolve relative to workspace root
    small_path = (workspace_root / small_model_path).resolve()
    large_path = (workspace_root / large_model_path).resolve()
    
    # Verify paths exist
    if not small_path.exists():
        pytest.skip(f"Small model checkpoint not found: {small_path}")
    if not large_path.exists():
        pytest.skip(f"Large model checkpoint not found: {large_path}")
    
    # Load random game states from default MCTS logs directory
    log_dir = (workspace_root / DEFAULT_MCTS_LOGS_DIR).resolve()
    
    if not log_dir.exists():
        pytest.skip(f"MCTS logs directory not found: {log_dir}")
    
    try:
        game_states = load_random_game_states(log_dir, num_states=20, seed=42)
    except ValueError as e:
        pytest.skip(str(e))
    
    # Initialize inference models - architecture constants are auto-detected
    try:
        small_inference = PolicySmallInference(small_path)
    except Exception as e:
        pytest.skip(f"Failed to load small model from checkpoint: {e}")
    
    try:
        large_inference = PolicyLargeInference(large_path)
    except Exception as e:
        pytest.skip(f"Failed to load large model from checkpoint: {e}")
    
    # Warm up both models (first inference can be slower)
    if game_states:
        _ = small_inference(game_states[0])
        _ = large_inference(game_states[0])
    
    # Measure inference times
    small_times = []
    large_times = []
    
    for state in game_states:
        # Measure small model inference
        start_time = time.perf_counter()
        _ = small_inference(state)
        small_time = time.perf_counter() - start_time
        small_times.append(small_time)
        
        # Measure large model inference
        start_time = time.perf_counter()
        _ = large_inference(state)
        large_time = time.perf_counter() - start_time
        large_times.append(large_time)
    
    # Calculate statistics
    small_avg = statistics.mean(small_times)
    small_std = statistics.stdev(small_times) if len(small_times) > 1 else 0.0
    large_avg = statistics.mean(large_times)
    large_std = statistics.stdev(large_times) if len(large_times) > 1 else 0.0
    
    speedup = large_avg / small_avg if small_avg > 0 else 0.0
    
    # Print results
    print(f"\n=== Inference Performance Comparison ===")
    print(f"Small Model: {small_path.name}")
    print(f"Large Model: {large_path.name}")
    print(f"Number of game states tested: {len(game_states)}")
    print(f"\nSmall Model:")
    print(f"  Average inference time: {small_avg*1000:.4f} ms")
    print(f"  Std deviation: {small_std*1000:.4f} ms")
    print(f"  Min: {min(small_times)*1000:.4f} ms")
    print(f"  Max: {max(small_times)*1000:.4f} ms")
    print(f"\nLarge Model:")
    print(f"  Average inference time: {large_avg*1000:.4f} ms")
    print(f"  Std deviation: {large_std*1000:.4f} ms")
    print(f"  Min: {min(large_times)*1000:.4f} ms")
    print(f"  Max: {max(large_times)*1000:.4f} ms")
    print(f"\nSpeedup (Large/Small): {speedup:.2f}x")
    print(f"Time difference: {(large_avg - small_avg)*1000:.4f} ms")
    
    # Assertions
    assert len(small_times) == 20, "Should have tested 20 game states"
    assert len(large_times) == 20, "Should have tested 20 game states"
    assert small_avg > 0, "Small model inference time should be positive"
    assert large_avg > 0, "Large model inference time should be positive"
    assert all(t > 0 for t in small_times), "All small model inference times should be positive"
    assert all(t > 0 for t in large_times), "All large model inference times should be positive"


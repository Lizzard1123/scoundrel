# RL interfaces

# Interactive Viewer System

The viewer system provides a standardized way to watch agents play Scoundrel interactively. All viewers share a common base implementation while allowing customization for agent-specific features.

## Base Viewer

The core viewer is `run_interactive_viewer()` in `scoundrel/rl/viewer.py`. It handles:
- Game state management
- User input (step, play, restart, quit)
- Autoplay with minimum delay
- Keyboard interrupt handling

### Usage

```python
from typing import Any
from scoundrel.rl.viewer import run_interactive_viewer
from scoundrel.models.game_state import Action, GameState

def get_action_fn(state: GameState) -> tuple[Action, Any]:
    """Return (action, extra_info) for the current state."""
    action = your_agent.select_action(state)
    extra_info = your_agent.get_stats()  # Optional: any additional data
    return action, extra_info

run_interactive_viewer(
    get_action_fn=get_action_fn,
    label="Your Agent",
    checkpoint_name="checkpoint.pt",  # Optional
    seed=42,  # Optional
    format_ui_text_fn=None,  # Optional: custom formatting
)
```

## Creating a Custom Viewer

All custom viewers follow this standard pattern:

### 1. Define Action Function

```python
from typing import Any
from scoundrel.models.game_state import Action, GameState

def get_your_action(state: GameState) -> tuple[Action, Any]:
    """Get action and return with any extra info."""
    action_idx = agent.select_action(state)
    action_enum = agent.translator.decode_action(action_idx)
    extra_info = agent.get_stats()  # Your custom data
    return action_enum, extra_info
```

### 2. Optional: Custom UI Formatting

```python
from typing import Any
from scoundrel.models.game_state import Action
from scoundrel.rl.utils import format_action

def format_your_ui_text(action: Action, extra_info: Any) -> str:
    """Format UI text with custom styling/stats."""
    action_text = format_action(action)
    stats = extra_info  # Unpack your extra_info
    return f"Next: [bold green]{action_text}[/bold green] | Stats: {stats}"
```

### 3. Call Base Viewer

```python
from scoundrel.rl.viewer import run_interactive_viewer
from scoundrel.models.game_state import GameState

def run_your_viewer(**kwargs):
    agent = YourAgent(**kwargs)
    
    def get_action(state: GameState) -> tuple[Action, Any]:
        return get_your_action(state)
    
    run_interactive_viewer(
        get_action_fn=get_action,
        label="Your Agent",
        seed=seed,
        format_ui_text_fn=format_your_ui_text,  # Optional
    )
```

## Examples

### MCTS Viewer (`scoundrel/rl/mcts/viewer.py`)
- Custom formatting with action statistics
- Shows visit counts and scores for each action

### PPO Viewer (`scoundrel/rl/transformer_mlp/viewer.py`)
- Loads checkpoint from file
- Uses default formatting (simple action display)

## Standard Pattern Checklist

✅ Use `run_interactive_viewer()` from base viewer  
✅ Define `get_action_fn(state) -> tuple[Action, Any]`  
✅ Optionally define `format_ui_text_fn(action, extra_info) -> str`  
✅ Include argument parsing and `main()` entry point  
✅ Keep agent-specific logic separate from viewer logic

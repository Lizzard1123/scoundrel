"""
MCTS Agent implementation for Scoundrel.
Implements the four phases: Selection, Expansion, Simulation, Backpropagation.
Supports root parallelization for improved performance.
"""
import random
import math
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from scoundrel.game.game_manager import GameManager
from scoundrel.game.game_logic import apply_action_to_state
from scoundrel.models.game_state import GameState, Action
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.mcts.mcts_node import MCTSNode
from scoundrel.rl.mcts.constants import (
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    USE_RANDOM_ROLLOUT,
    MCTS_NUM_WORKERS,
)


def _run_worker_simulations(args: tuple) -> Dict:
    """
    Worker function for parallel MCTS simulations.
    Runs a batch of simulations and returns action statistics.
    
    This is a module-level function (not a method) so it can be pickled
    for multiprocessing.
    
    Args:
        args: Tuple of (game_state, num_simulations, worker_id, exploration_constant, 
                        max_depth, use_random_rollout)
        
    Returns:
        Dictionary with action statistics: {action: {visits, value}}
    """
    game_state, num_simulations, worker_id, exploration_constant, max_depth, use_random_rollout = args
    
    # Create a new agent for this worker (each worker is independent)
    agent = MCTSAgent(
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
        max_depth=max_depth,
        use_random_rollout=use_random_rollout,
        num_workers=0  # Disable parallelization in worker
    )
    
    # Run sequential search
    root = agent._sequential_search(game_state)
    
    # Extract action statistics from root's children
    action_stats = {}
    for child in root.children:
        action_stats[child.action] = {
            'visits': child.visits,
            'value': child.value
        }
    
    return {
        'action_stats': action_stats,
        'worker_id': worker_id,
        'root_visits': root.visits
    }


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for Scoundrel.
    """
    
    def __init__(
        self,
        num_simulations: int = 100,
        exploration_constant: float = MCTS_EXPLORATION_CONSTANT,
        max_depth: int = MCTS_MAX_DEPTH,
        use_random_rollout: bool = USE_RANDOM_ROLLOUT,
        num_workers: int = MCTS_NUM_WORKERS,
    ):
        """
        Initialize MCTS agent.
        
        Args:
            num_simulations: Number of simulations to run per move
            exploration_constant: UCB1 exploration constant
            max_depth: Maximum depth for simulation rollouts
            use_random_rollout: Whether to use random policy for rollouts
            num_workers: Number of parallel workers (0 or 1 disables parallelization)
        """
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.use_random_rollout = use_random_rollout
        self.num_workers = num_workers
        self.translator = ScoundrelTranslator()
    
    def select_action(self, game_state: GameState) -> int:
        """
        Select the best action using MCTS.
        Uses parallel execution if num_workers > 1.
        
        Args:
            game_state: Current game state
            
        Returns:
            Action index (0-4)
        """
        # Determine if parallelization should be used
        use_parallel = self.num_workers > 1
        
        if use_parallel:
            root = self._parallel_search(game_state)
        else:
            root = self._sequential_search(game_state)
        
        # Store root for get_action_stats
        self._last_root = root
        
        # Select action with most visits
        if not root.children:
            # No simulations completed, return random valid action
            valid_actions = self._get_valid_actions(game_state)
            return random.choice(valid_actions) if valid_actions else 0
        
        best_child = root.most_visited_child()
        return best_child.action
    
    def _sequential_search(self, game_state: GameState) -> MCTSNode:
        """
        Run MCTS simulations sequentially.
        
        Args:
            game_state: Current game state
            
        Returns:
            Root node with simulation results
        """
        root = self._create_node(game_state)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Start from root state with determinization
            # This handles hidden information: shuffles unknown cards, keeps known cards
            simulation_state = self._determinize_state(game_state)
            
            # 1. Selection: Traverse tree to find node to expand
            node, simulation_state = self._select(root, simulation_state)
            
            # 2. Expansion: Add new child node if not terminal
            if not simulation_state.game_over and not node.is_fully_expanded():
                node, simulation_state = self._expand(node, simulation_state)
            
            # 3. Simulation: Play out from this node to get reward
            reward = self._simulate(simulation_state)
            
            # 4. Backpropagation: Update all nodes in path
            self._backpropagate(node, reward)
        
        return root
    
    def _parallel_search(self, game_state: GameState) -> MCTSNode:
        """
        Run MCTS simulations in parallel using root parallelization.
        Each worker runs independent simulations, then results are aggregated.
        
        Args:
            game_state: Current game state
            
        Returns:
            Aggregated root node with combined simulation results
        """
        # Ensure we use 'spawn' method for multiprocessing (required on macOS)
        ctx = mp.get_context('spawn')
        
        # Divide simulations among workers
        sims_per_worker = self.num_simulations // self.num_workers
        remaining_sims = self.num_simulations % self.num_workers
        
        # Create worker arguments - include all parameters needed to recreate agent
        worker_args = []
        for i in range(self.num_workers):
            # Distribute remaining simulations to first workers
            num_sims = sims_per_worker + (1 if i < remaining_sims else 0)
            worker_args.append((
                game_state, 
                num_sims, 
                i,
                self.exploration_constant,
                self.max_depth,
                self.use_random_rollout
            ))
        
        # Run workers in parallel using spawn context
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx) as executor:
            worker_results = list(executor.map(
                _run_worker_simulations,
                worker_args
            ))
        
        # Aggregate results from all workers
        return self._aggregate_roots(worker_results, game_state)
    
    def _aggregate_roots(
        self, 
        worker_results: List[Dict], 
        game_state: GameState
    ) -> MCTSNode:
        """
        Aggregate results from multiple worker trees into a single root node.
        Combines visit counts and values for each action.
        
        Args:
            worker_results: List of dictionaries with action statistics from each worker
            game_state: Original game state (for creating aggregated root)
            
        Returns:
            Root node with aggregated statistics
        """
        # Create aggregated root node
        root = self._create_node(game_state)
        root.untried_actions = []  # Mark as fully expanded
        
        # Collect all actions seen across workers
        action_stats = {}  # action -> {visits, value}
        
        for worker_result in worker_results:
            for action, stats in worker_result['action_stats'].items():
                if action not in action_stats:
                    action_stats[action] = {'visits': 0, 'value': 0.0}
                action_stats[action]['visits'] += stats['visits']
                action_stats[action]['value'] += stats['value']
        
        # Create child nodes for each action with aggregated stats
        for action, stats in action_stats.items():
            # Create a dummy state for the child (not used for action selection)
            child = MCTSNode(
                state_hash=f"aggregated_child_{action}",
                parent=root,
                action=action,
                visits=stats['visits'],
                value=stats['value']
            )
            root.children.append(child)
        
        # Update root visits (sum of all child visits)
        root.visits = sum(child.visits for child in root.children)
        
        return root
    
    def get_action_stats(self, game_state: GameState):
        """
        Get statistics for all valid actions from the last MCTS search.
        Must be called after select_action().
        
        Returns:
            List of (action_idx, visits, avg_value, ucb1_score) tuples
        """
        if not hasattr(self, '_last_root') or not self._last_root.children:
            return []
        
        stats = []
        for child in self._last_root.children:
            avg_value = child.value / child.visits if child.visits > 0 else 0
            # Calculate UCB1 (same as used during search)
            if child.visits > 0 and self._last_root.visits > 0:
                ucb1 = avg_value + self.exploration_constant * math.sqrt(
                    math.log(self._last_root.visits) / child.visits
                )
            else:
                ucb1 = float('inf')
            
            stats.append({
                'action': child.action,
                'visits': child.visits,
                'avg_value': avg_value,
                'total_value': child.value,
                'ucb1': ucb1
            })
        
        # Sort by action index for consistent display
        stats.sort(key=lambda x: x['action'])
        return stats
    
    def _create_node(
        self,
        game_state: GameState,
        parent: Optional[MCTSNode] = None,
        action: Optional[int] = None
    ) -> MCTSNode:
        """Create a new MCTS node from game state."""
        state_hash = self._hash_state(game_state)
        valid_actions = self._get_valid_actions(game_state)
        
        return MCTSNode(
            state_hash=state_hash,
            parent=parent,
            action=action,
            untried_actions=valid_actions,
            is_game_over=game_state.game_over
        )
    
    def _hash_state(self, game_state: GameState) -> str:
        """Create a hash representing the game state."""
        # Simple hash based on key state features
        room_str = ",".join([f"{c.suit.value}{c.value}" for c in game_state.room])
        dungeon_str = ",".join([f"{c.suit.value}{c.value}" for c in game_state.dungeon[:5]])  # First 5 cards
        weapon_str = f"{game_state.equipped_weapon.value}" if game_state.equipped_weapon else "None"
        return f"h{game_state.health}_w{weapon_str}_r{room_str}_d{dungeon_str}"
    
    def _get_valid_actions(self, game_state: GameState) -> List[int]:
        """Get list of valid action indices for current state."""
        mask = self.translator.get_action_mask(game_state)
        return [i for i, valid in enumerate(mask) if valid]
    
    def _apply_action_to_state(self, game_state: GameState, action: Action) -> GameState:
        """
        Apply an action to a game state and return a new state.
        
        This eliminates the need for engine creation during selection and expansion.
        Uses the shared pure function from game_logic module.
        
        Args:
            game_state: Current game state (will be copied, not mutated)
            action: Action to apply
            
        Returns:
            New GameState with the action applied
        """
        return apply_action_to_state(game_state, action)
    
    def _select(self, node: MCTSNode, game_state: GameState) -> tuple[MCTSNode, GameState]:
        """
        Selection phase: Traverse tree using UCB1 until we find a node to expand.
        Returns the selected node and its corresponding game state.
        """
        current_node = node
        current_state = game_state
        
        while current_node.is_fully_expanded() and current_node.children and not current_state.game_over:
            current_node = current_node.best_child(self.exploration_constant)
            # Apply action to state using pure function (no engine creation)
            action_enum = self.translator.decode_action(current_node.action)
            current_state = self._apply_action_to_state(current_state, action_enum)
        
        return current_node, current_state
    
    def _expand(self, node: MCTSNode, game_state: GameState) -> tuple[MCTSNode, GameState]:
        """
        Expansion phase: Add a new child node for an untried action.
        Returns the new child node and its corresponding game state.
        """
        if len(node.untried_actions) == 0:
            return node, game_state
        
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Apply action to get new state using pure function (no engine creation)
        action_enum = self.translator.decode_action(action)
        new_state = self._apply_action_to_state(game_state, action_enum)
        
        # Create new child node
        child_node = self._create_node(new_state, parent=node, action=action)
        node.children.append(child_node)
        
        return child_node, new_state
    
    def _simulate(self, game_state: GameState) -> float:
        """
        Simulation phase: Play out the game using rollout policy.
        Returns normalized reward.
        """
        engine = self._create_engine_from_state(game_state)
        current_state = engine.get_state()
        
        depth = 0
        while not current_state.game_over and depth < self.max_depth:
            # Get valid actions
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Select action using rollout policy
            if self.use_random_rollout:
                action = self._random_policy(current_state)
            else:
                action = self._heuristic_policy(current_state)
            
            # Apply action
            action_enum = self.translator.decode_action(action)
            engine.execute_turn(action_enum)
            current_state = engine.get_state()
            depth += 1
        
        # Return normalized reward
        return self._normalize_reward(current_state.score)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update all nodes in path with simulation result.
        """
        current_node = node
        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent
    
    def _random_policy(self, game_state: GameState) -> int:
        """Random action selection for rollout."""
        valid_actions = self._get_valid_actions(game_state)
        return random.choice(valid_actions)
    
    def _heuristic_policy(self, game_state: GameState) -> int:
        """Heuristic action selection for rollout based on Scoundrel mechanics."""
        from scoundrel.models.card import CardType
        from scoundrel.game.combat import Combat
        
        valid_actions = self._get_valid_actions(game_state)
        if not valid_actions:
            return 0
        
        # If we can avoid and health is low, consider avoiding
        if game_state.can_avoid and game_state.health < 8:
            return 4  # Avoid
        
        # Evaluate each card in the room
        best_action = None
        best_score = float('-inf')
        
        for action_idx in valid_actions:
            if action_idx == 4:  # Avoid action
                score = -5  # Slightly discourage avoiding unless necessary
            else:
                card = game_state.room[action_idx]
                score = 0
                
                if card.type == CardType.POTION:
                    # Value potions based on how much health we need
                    health_needed = 20 - game_state.health
                    if health_needed > 0 and game_state.can_use_potion:
                        score = min(card.value, health_needed) * 2  # High value
                    else:
                        score = -10  # Don't take if we can't use it
                
                elif card.type == CardType.WEAPON:
                    # Value weapons based on their strength
                    if game_state.equipped_weapon:
                        # Upgrade if new weapon is better
                        score = (card.value - game_state.equipped_weapon.value) * 2
                    else:
                        score = card.value * 3  # High value for first weapon
                
                elif card.type == CardType.MONSTER:
                    # Calculate damage we'd take
                    if Combat.can_use_weapon(game_state, card):
                        damage = Combat.calculate_damage(card, game_state.equipped_weapon)
                    else:
                        damage = card.value
                    
                    # Negative score based on damage
                    score = -damage * 2
                    
                    # Avoid if it would kill us
                    if damage >= game_state.health:
                        score = -1000
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action if best_action is not None else random.choice(valid_actions)
    
    def _determinize_state(self, game_state: GameState) -> GameState:
        """
        Apply determinization to handle hidden information.
        Shuffles unknown cards in the dungeon while keeping known cards in place.
        
        In Scoundrel, when you avoid a room, those 4 cards go to the bottom of the dungeon.
        The agent knows their position (they're at indices >= number_avoided * 4).
        The cards at the top (indices < number_avoided * 4) are unknown and should be shuffled.
        """
        determinized_state = game_state.copy()
        
        # Shuffle the unknown portion of the dungeon
        if determinized_state.dungeon and determinized_state.number_avoided > 0:
            unknown_count = determinized_state.number_avoided * 4
            if unknown_count < len(determinized_state.dungeon):
                # Split dungeon into unknown and known parts
                unknown_cards = determinized_state.dungeon[:unknown_count]
                known_cards = determinized_state.dungeon[unknown_count:]
                
                # Shuffle only the unknown cards
                random.shuffle(unknown_cards)
                
                # Reconstruct dungeon with shuffled unknown + preserved known
                determinized_state.dungeon = unknown_cards + known_cards
        
        return determinized_state
    
    def _create_engine_from_state(self, game_state: GameState) -> GameManager:
        """
        Create a lightweight GameManager instance with the given state.
        
        Uses GameManager.from_state() to bypass expensive initialization:
        - No random seed generation
        - No TerminalUI creation
        - No deck creation (state already contains the deck)
        
        This provides significant performance improvements when called
        thousands of times during MCTS simulations.
        """
        return GameManager.from_state(game_state.copy())
    
    def _normalize_reward(self, score: int) -> float:
        """
        Normalize game score to [0, 1] range.
        
        Score bounds:
        - Min: -188 (die at 0 HP with max monsters remaining)
          It takes exactly 20 damage to die, leaving 208-20=188 monster value
          Score = 0 - 188 = -188
        - Max: 30 (20 HP + max potion = 10)
        - Range: 218
        """
        min_score = -188
        max_score = 30
        return (score - min_score) / (max_score - min_score)


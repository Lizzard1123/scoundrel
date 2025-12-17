"""
MCTS Agent implementation for Scoundrel.
Implements the four phases: Selection, Expansion, Simulation, Backpropagation.
Supports root parallelization for improved performance.
"""
import random
import math
from typing import List, Optional, Dict
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from scoundrel.game.game_manager import GameManager
from scoundrel.game.game_logic import apply_action_to_state
from scoundrel.models.game_state import GameState, Action
from scoundrel.models.card import Suit, CardType
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.mcts.mcts_node import MCTSNode
from scoundrel.rl.mcts.constants import (
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    USE_RANDOM_ROLLOUT,
    MCTS_NUM_WORKERS,
    MCTS_TRANSPOSITION_TABLE_SIZE,
)

# Suit to index mapping for efficient hashing
# HEARTS=0, DIAMONDS=1, CLUBS=2, SPADES=3
_SUIT_TO_INDEX = {suit: idx for idx, suit in enumerate(Suit)}


class TranspositionTable:
    """
    LRU cache for MCTS state evaluations.
    Caches simulation results (normalized rewards) by state hash.
    
    Uses OrderedDict for O(1) access and LRU eviction.
    """
    
    def __init__(self, max_size: int = MCTS_TRANSPOSITION_TABLE_SIZE):
        """
        Initialize transposition table.
        
        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
        """
        self.max_size = max_size
        self.cache: OrderedDict[int, float] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, state_hash: int) -> Optional[float]:
        """
        Get cached reward for a state hash.
        
        Args:
            state_hash: Hash of the game state
            
        Returns:
            Cached normalized reward if found, None otherwise
        """
        if state_hash in self.cache:
            # Move to end (most recently used)
            reward = self.cache.pop(state_hash)
            self.cache[state_hash] = reward
            self.hits += 1
            return reward
        
        self.misses += 1
        return None
    
    def put(self, state_hash: int, reward: float):
        """
        Store reward for a state hash.
        
        Args:
            state_hash: Hash of the game state
            reward: Normalized reward from simulation
        """
        if state_hash in self.cache:
            # Update existing entry and move to end
            self.cache.pop(state_hash)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used (first item)
            self.cache.popitem(last=False)
        
        self.cache[state_hash] = reward
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, size, and hit_rate
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate
        }


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
    
    # Get cache statistics from worker
    cache_stats = agent.get_cache_stats()
    
    return {
        'action_stats': action_stats,
        'worker_id': worker_id,
        'root_visits': root.visits,
        'cache_stats': cache_stats
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
        max_cache_size: int = MCTS_TRANSPOSITION_TABLE_SIZE,
    ):
        """
        Initialize MCTS agent.
        
        Args:
            num_simulations: Number of simulations to run per move
            exploration_constant: UCB1 exploration constant
            max_depth: Maximum depth for simulation rollouts
            use_random_rollout: Whether to use random policy for rollouts
            num_workers: Number of parallel workers (0 or 1 disables parallelization)
            max_cache_size: Maximum size of transposition table (LRU eviction)
        """
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.use_random_rollout = use_random_rollout
        self.num_workers = num_workers
        self.translator = ScoundrelTranslator()
        self.transposition_table = TranspositionTable(max_size=max_cache_size)
    
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
        # Note: We don't reset accumulated stats here - they accumulate across moves
        # Cache size is now accumulated, not reset per move
        
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
        
        # Aggregate cache statistics from all workers
        self._aggregate_cache_stats(worker_results)
        
        # Aggregate results from all workers
        return self._aggregate_roots(worker_results, game_state)
    
    def _aggregate_cache_stats(self, worker_results: List[Dict]):
        """
        Aggregate cache statistics from all worker processes.
        Accumulates statistics across multiple moves (workers terminate after each move).
        
        Args:
            worker_results: List of dictionaries with results from each worker
        """
        total_hits = 0
        total_misses = 0
        total_cache_size = 0
        
        for worker_result in worker_results:
            if 'cache_stats' in worker_result:
                cache_stats = worker_result['cache_stats']
                total_hits += cache_stats.get('hits', 0)
                total_misses += cache_stats.get('misses', 0)
                total_cache_size += cache_stats.get('size', 0)
        
        # Accumulate statistics across all moves (don't reset, accumulate)
        # Note: In parallel mode, workers terminate after each move, so we accumulate
        # stats across moves to get total statistics for the entire game/session
        if not hasattr(self, '_accumulated_hits'):
            self._accumulated_hits = 0
            self._accumulated_misses = 0
            self._accumulated_cache_size = 0
        
        self._accumulated_hits += total_hits
        self._accumulated_misses += total_misses
        # Accumulate cache size across moves (sum of all unique states seen across all moves)
        # Note: This is an approximation - actual unique states might be less due to overlaps
        # between workers and moves, but it gives a sense of total cache usage
        self._accumulated_cache_size += total_cache_size
        
        # Update main agent's cache statistics to reflect accumulated values
        self.transposition_table.hits = self._accumulated_hits
        self.transposition_table.misses = self._accumulated_misses
        # Clear the main cache dict (we're tracking aggregate stats, not actual cache contents)
        self.transposition_table.cache.clear()
        # Set aggregated_cache_size for get_cache_stats() to use
        self._aggregated_cache_size = self._accumulated_cache_size
    
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
            # Create a dummy state hash for the child (not used for action selection)
            # Use a deterministic integer hash based on action
            child_hash = hash(("aggregated_child", action))
            child = MCTSNode(
                state_hash=child_hash,
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
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get transposition table cache statistics.
        
        For parallel execution, returns aggregated statistics from all workers.
        For sequential execution, returns statistics from the main agent's cache.
        
        Returns:
            Dictionary with hits, misses, size, max_size, and hit_rate
        """
        stats = self.transposition_table.stats()
        # If we have aggregated cache size from parallel workers, use it
        if hasattr(self, '_aggregated_cache_size'):
            stats['size'] = self._aggregated_cache_size
            # Max size is per-worker, so total max is num_workers * max_size
            stats['max_size'] = self.num_workers * self.transposition_table.max_size if self.num_workers > 1 else self.transposition_table.max_size
        return stats
    
    def clear_cache(self):
        """Clear the transposition table cache."""
        self.transposition_table.clear()
    
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
    
    def _hash_state(self, game_state: GameState) -> int:
        """
        Create an integer hash representing the game state.
        
        Uses efficient integer-based hashing instead of string concatenation.
        Converts cards to integers: suit_index * 100 + card_value
        Then uses Python's built-in hash() on a tuple of state features.
        
        Args:
            game_state: Game state to hash
            
        Returns:
            Integer hash of the game state
        """
        # Helper function to convert card to integer
        # Uses suit_index * 100 + card_value to ensure unique integer per card
        def card_to_int(card):
            suit_index = _SUIT_TO_INDEX[card.suit]
            return suit_index * 100 + card.value
        
        # Convert room cards to integers
        room_tuple = tuple(card_to_int(c) for c in game_state.room)
        
        # Convert first 5 dungeon cards to integers
        dungeon_cards = game_state.dungeon[:5]
        dungeon_tuple = tuple(card_to_int(c) for c in dungeon_cards)
        
        # Weapon value: card.value if equipped, else 0
        weapon_value = game_state.equipped_weapon.value if game_state.equipped_weapon else 0
        
        # Create tuple of key state features and hash it
        state_tuple = (game_state.health, weapon_value, room_tuple, dungeon_tuple)
        return hash(state_tuple)
    
    def _get_valid_actions(self, game_state: GameState) -> List[int]:
        """
        Get list of valid action indices for current state.
        Uses inline validation to avoid tensor creation overhead.
        
        Actions:
        - 0-3: Pick card from room (valid if index < len(room))
        - 4: Avoid room (valid if can_avoid is True)
        """
        valid_actions = []
        
        # Pick actions (0-3): valid if room has a card at that index
        for i in range(len(game_state.room)):
            valid_actions.append(i)
        
        # Avoid action (4): valid if can_avoid is True
        if game_state.can_avoid:
            valid_actions.append(4)
        
        return valid_actions
    
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
        
        Uses transposition table to cache simulation results for identical states.
        """
        # Check transposition table for cached result
        state_hash = self._hash_state(game_state)
        cached_reward = self.transposition_table.get(state_hash)
        if cached_reward is not None:
            return cached_reward
        
        # Run simulation
        engine = self._create_engine_from_state(game_state)
        current_state = engine.get_state()
        
        depth = 0
        while not current_state.game_over and depth < self.max_depth:
            # Get valid actions
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Select action using rollout policy
            # Pass valid_actions to avoid recomputation in policy methods
            if self.use_random_rollout:
                action = self._random_policy(current_state, valid_actions)
            else:
                action = self._heuristic_policy(current_state, valid_actions)
            
            # Apply action
            action_enum = self.translator.decode_action(action)
            engine.execute_turn(action_enum)
            current_state = engine.get_state()
            depth += 1
        
        # Calculate and cache normalized reward
        reward = self._normalize_reward(current_state.score)
        self.transposition_table.put(state_hash, reward)
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update all nodes in path with simulation result.
        """
        current_node = node
        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent
    
    def _random_policy(self, game_state: GameState, valid_actions: List[int]) -> int:
        """
        Random action selection for rollout.
        
        Args:
            game_state: Current game state (unused, kept for API consistency)
            valid_actions: Pre-computed list of valid action indices
        """
        return random.choice(valid_actions)
    
    def _heuristic_policy(self, game_state: GameState, valid_actions: List[int]) -> int:
        """
        Heuristic action selection for rollout based on Scoundrel mechanics.
        Optimized to avoid redundant computations and inline simple logic.
        
        Args:
            game_state: Current game state
            valid_actions: Pre-computed list of valid action indices
        """
        if not valid_actions:
            return 0
        
        # Early exit: If we can avoid and health is low, prefer avoiding
        if game_state.can_avoid and game_state.health < 8:
            return 4  # Avoid
        
        # Evaluate each card in the room
        best_action = None
        best_score = float('-inf')
        
        # Pre-compute common values to avoid repeated property access
        health = game_state.health
        equipped_weapon = game_state.equipped_weapon
        can_use_potion = game_state.can_use_potion
        weapon_monsters = game_state.weapon_monsters
        
        for action_idx in valid_actions:
            if action_idx == 4:  # Avoid action
                score = -5  # Slightly discourage avoiding unless necessary
            else:
                card = game_state.room[action_idx]
                card_type = card.type
                
                if card_type == CardType.POTION:
                    # Value potions based on how much health we need
                    health_needed = 20 - health
                    if health_needed > 0 and can_use_potion:
                        score = min(card.value, health_needed) * 2  # High value
                    else:
                        score = -10  # Don't take if we can't use it
                
                elif card_type == CardType.WEAPON:
                    # Value weapons based on their strength
                    if equipped_weapon:
                        # Upgrade if new weapon is better
                        score = (card.value - equipped_weapon.value) * 2
                    else:
                        score = card.value * 3  # High value for first weapon
                
                elif card_type == CardType.MONSTER:
                    # Inline combat calculations to avoid function call overhead
                    # Equivalent to: Combat.can_use_weapon(game_state, card)
                    can_use_weapon = (
                        equipped_weapon is not None and
                        (not weapon_monsters or card.value <= weapon_monsters[-1].value)
                    )
                    
                    # Calculate damage (equivalent to Combat.calculate_damage)
                    if can_use_weapon:
                        # damage = max(0, monster.value - weapon.value)
                        damage = max(0, card.value - equipped_weapon.value)
                    else:
                        damage = card.value
                    
                    # Early exit: Avoid if it would kill us
                    if damage >= health:
                        score = -1000
                    else:
                        # Negative score based on damage
                        score = -damage * 2
            
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


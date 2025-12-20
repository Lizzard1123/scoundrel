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

from scoundrel.game.game_logic import apply_action_to_state
from scoundrel.models.game_state import GameState, Action
from scoundrel.models.card import Suit, CardType
from scoundrel.game.combat import Combat
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.mcts.mcts_node import MCTSNode
from scoundrel.rl.utils import normalize_score
from scoundrel.rl.mcts.constants import (
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    USE_RANDOM_ROLLOUT,
    MCTS_NUM_WORKERS,
    MCTS_TRANSPOSITION_TABLE_SIZE,
)

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
            self.cache.pop(state_hash)
        elif len(self.cache) >= self.max_size:
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
    
    agent = MCTSAgent(
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
        max_depth=max_depth,
        use_random_rollout=use_random_rollout,
        num_workers=0
    )
    
    root = agent._sequential_search(game_state)
    
    action_stats = {}
    for child in root.children:
        action_stats[child.action] = {
            'visits': child.visits,
            'value': child.value
        }
    
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
        use_parallel = self.num_workers > 1
        
        if use_parallel:
            root = self._parallel_search(game_state)
        else:
            root = self._sequential_search(game_state)
        
        self._last_root = root
        
        if not root.children:
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
        
        for _ in range(self.num_simulations):
            simulation_state = self._determinize_state(game_state)
            
            node, simulation_state = self._select(root, simulation_state)
            
            if not simulation_state.game_over and not node.is_fully_expanded():
                node, simulation_state = self._expand(node, simulation_state)
            
            if simulation_state.game_over:
                state_hash = self._hash_state(simulation_state)
                cached_reward = self.transposition_table.get(state_hash)
                if cached_reward is not None:
                    reward = cached_reward
                else:
                    reward = normalize_score(simulation_state.score)
                    self.transposition_table.put(state_hash, reward)
            else:
                reward = self._simulate(simulation_state)
            
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
        ctx = mp.get_context('spawn')
        
        sims_per_worker = self.num_simulations // self.num_workers
        remaining_sims = self.num_simulations % self.num_workers
        
        worker_args = []
        for i in range(self.num_workers):
            num_sims = sims_per_worker + (1 if i < remaining_sims else 0)
            worker_args.append((
                game_state, 
                num_sims, 
                i,
                self.exploration_constant,
                self.max_depth,
                self.use_random_rollout
            ))
        
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx) as executor:
            worker_results = list(executor.map(
                _run_worker_simulations,
                worker_args
            ))
        
        self._aggregate_cache_stats(worker_results)
        
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
        
        if not hasattr(self, '_accumulated_hits'):
            self._accumulated_hits = 0
            self._accumulated_misses = 0
            self._accumulated_cache_size = 0
        
        self._accumulated_hits += total_hits
        self._accumulated_misses += total_misses
        self._accumulated_cache_size += total_cache_size
        
        self.transposition_table.hits = self._accumulated_hits
        self.transposition_table.misses = self._accumulated_misses
        self.transposition_table.cache.clear()
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
        root = self._create_node(game_state)
        root.untried_actions = []
        
        action_stats = {}
        
        for worker_result in worker_results:
            for action, stats in worker_result['action_stats'].items():
                if action not in action_stats:
                    action_stats[action] = {'visits': 0, 'value': 0.0}
                action_stats[action]['visits'] += stats['visits']
                action_stats[action]['value'] += stats['value']
        
        for action, stats in action_stats.items():
            child_hash = hash(("aggregated_child", action))
            child = MCTSNode(
                state_hash=child_hash,
                parent=root,
                action=action,
                visits=stats['visits'],
                value=stats['value']
            )
            root.children.append(child)
        
        root.visits = sum(child.visits for child in root.children)
        
        return root
    
    def get_action_stats(self):
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
        if hasattr(self, '_aggregated_cache_size'):
            stats['size'] = self._aggregated_cache_size
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
        
        IMPORTANT: Must include last_used_card in hash since it affects score calculation
        (bonus points depend on whether last card was a potion).
        
        Args:
            game_state: Game state to hash
            
        Returns:
            Integer hash of the game state
        """
        def card_to_int(card):
            suit_index = _SUIT_TO_INDEX[card.suit]
            return suit_index * 100 + card.value
        
        room_tuple = tuple(card_to_int(c) for c in game_state.room)
        
        dungeon_cards = game_state.dungeon[:5]
        dungeon_tuple = tuple(card_to_int(c) for c in dungeon_cards)
        
        weapon_value = game_state.equipped_weapon.value if game_state.equipped_weapon else 0
        
        # Include last_used_card in hash since it affects score (bonus points)
        last_card_int = card_to_int(game_state.last_used_card) if game_state.last_used_card else 0
        
        state_tuple = (game_state.health, weapon_value, room_tuple, dungeon_tuple, last_card_int)
        return hash(state_tuple)
    
    def _get_valid_actions(self, game_state: GameState) -> List[int]:
        """Get list of valid action indices for current state."""
        valid_actions = []
        
        for i in range(len(game_state.room)):
            valid_actions.append(i)
        
        if game_state.can_avoid:
            valid_actions.append(4)
        
        return valid_actions
    
    def _select(self, node: MCTSNode, game_state: GameState) -> tuple[MCTSNode, GameState]:
        """
        Selection phase: Traverse tree using UCB1 until we find a node to expand.
        Returns the selected node and its corresponding game state.
        """
        current_node = node
        current_state = game_state
        
        while current_node.is_fully_expanded() and current_node.children and not current_state.game_over:
            current_node = current_node.best_child(self.exploration_constant)
            action_enum = self.translator.decode_action(current_node.action)
            current_state = apply_action_to_state(current_state, action_enum)
        
        return current_node, current_state
    
    def _expand(self, node: MCTSNode, game_state: GameState) -> tuple[MCTSNode, GameState]:
        """Expansion phase: Add a new child node for an untried action."""
        if game_state.game_over:
            return node, game_state
        
        if len(node.untried_actions) == 0:
            return node, game_state
        
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        action_enum = self.translator.decode_action(action)
        new_state = apply_action_to_state(game_state, action_enum)
        
        child_node = self._create_node(new_state, parent=node, action=action)
        node.children.append(child_node)
        
        return child_node, new_state
    
    def _simulate(self, game_state: GameState) -> float:
        """
        Simulation phase: Play out the game using rollout policy.
        Returns normalized reward.
        
        Uses transposition table to cache simulation results for identical states.
        """
        state_hash = self._hash_state(game_state)
        cached_reward = self.transposition_table.get(state_hash)
        if cached_reward is not None:
            return cached_reward
        
        if game_state.game_over:
            reward = normalize_score(game_state.score)
            self.transposition_table.put(state_hash, reward)
            return reward
        
        current_state = game_state
        
        depth = 0
        while depth < self.max_depth:
            if current_state.game_over:
                break
            
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
            
            if self.use_random_rollout:
                action = self._random_policy(current_state, valid_actions)
            else:
                action = self._heuristic_policy(current_state, valid_actions)
            
            action_enum = self.translator.decode_action(action)
            current_state = apply_action_to_state(current_state, action_enum)
            depth += 1
            
            if current_state.game_over:
                break
        
        reward = normalize_score(current_state.score)
        self.transposition_table.put(state_hash, reward)
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: Update all nodes in path with simulation result."""
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
        
        if game_state.can_avoid and game_state.health < 8:
            return 4
        
        best_action = None
        best_score = float('-inf')
        
        health = game_state.health
        equipped_weapon = game_state.equipped_weapon
        can_use_potion = game_state.can_use_potion
        weapon_monsters = game_state.weapon_monsters
        
        for action_idx in valid_actions:
            if action_idx == 4:
                score = -5
            else:
                card = game_state.room[action_idx]
                card_type = card.type
                
                if card_type == CardType.POTION:
                    health_needed = 20 - health
                    if health_needed > 0 and can_use_potion:
                        score = min(card.value, health_needed) * 2
                    else:
                        score = -10
                
                elif card_type == CardType.WEAPON:
                    if equipped_weapon:
                        score = (card.value - equipped_weapon.value) * 2
                    else:
                        score = card.value * 3
                
                elif card_type == CardType.MONSTER:
                    can_use_weapon = Combat.can_use_weapon(game_state, card)
                    damage = Combat.calculate_damage(card, equipped_weapon)
                    
                    if damage >= health:
                        score = -1000
                    else:
                        score = -damage * 2
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action if best_action is not None else random.choice(valid_actions)
    
    def _determinize_state(self, game_state: GameState) -> GameState:
        """
        Apply determinization to handle hidden information.
        Shuffles unknown cards in the dungeon while keeping known cards in place
        """
        if game_state.number_avoided == 0 or not game_state.dungeon:
            return game_state.copy()
        
        determinized_state = game_state.copy()
        
        # Known cards = avoided rooms * 4 cards per room (at BACK of dungeon)
        known_count = determinized_state.number_avoided * 4
        
        if known_count < len(determinized_state.dungeon):
            # Shuffle unknown cards at front, keep known cards at back fixed
            unknown_cards = determinized_state.dungeon[:-known_count]
            known_cards = determinized_state.dungeon[-known_count:]
            random.shuffle(unknown_cards)
            
            determinized_state.dungeon = unknown_cards + known_cards
        # else: all cards are known (avoided), don't shuffle
        
        return determinized_state


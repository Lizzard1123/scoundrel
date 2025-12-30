"""
AlphaGo-style MCTS Agent implementation for Scoundrel.
Combines neural networks (PolicyLarge, PolicySmall, ValueLarge) with MCTS.

Key enhancements over vanilla MCTS:
1. PUCT formula uses policy priors from PolicyLarge
2. Value network + fast rollout for leaf evaluation
3. PolicySmall for fast rollouts (1 FC layer vs 10)
"""
import random
import math
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import torch

from scoundrel.game.game_logic import apply_action_to_state
from scoundrel.models.game_state import GameState, Action
from scoundrel.models.card import Suit
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_node import AlphaGoNode
from scoundrel.rl.utils import normalize_score
from scoundrel.rl.alpha_scoundrel.alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_C_PUCT,
    ALPHAGO_VALUE_WEIGHT,
    ALPHAGO_MAX_DEPTH,
    ALPHAGO_NUM_WORKERS,
    ALPHAGO_TRANSPOSITION_TABLE_SIZE,
    POLICY_LARGE_CHECKPOINT,
    POLICY_SMALL_CHECKPOINT,
    VALUE_LARGE_CHECKPOINT,
)
from scoundrel.rl.alpha_scoundrel.policy.policy_large.inference import PolicyLargeInference
from scoundrel.rl.alpha_scoundrel.policy.policy_small.inference import PolicySmallInference
from scoundrel.rl.alpha_scoundrel.value.value_large.inference import ValueLargeInference

_SUIT_TO_INDEX = {suit: idx for idx, suit in enumerate(Suit)}


class TranspositionTable:
    """
    LRU cache for AlphaGo MCTS state evaluations.
    Caches both value network predictions and rollout results.
    
    Uses OrderedDict for O(1) access and LRU eviction.
    """
    
    def __init__(self, max_size: int = ALPHAGO_TRANSPOSITION_TABLE_SIZE):
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
        Get cached value for a state hash.
        
        Args:
            state_hash: Hash of the game state
            
        Returns:
            Cached normalized value if found, None otherwise
        """
        if state_hash in self.cache:
            value = self.cache.pop(state_hash)
            self.cache[state_hash] = value
            self.hits += 1
            return value
        
        self.misses += 1
        return None
    
    def put(self, state_hash: int, value: float):
        """
        Store value for a state hash.
        
        Args:
            state_hash: Hash of the game state
            value: Normalized value from evaluation
        """
        if state_hash in self.cache:
            self.cache.pop(state_hash)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[state_hash] = value
    
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


class AlphaGoAgent:
    """
    AlphaGo-style MCTS agent for Scoundrel.
    
    Combines three neural networks with MCTS:
    1. PolicyLarge - Strategic policy priors P(s,a) for PUCT
    2. PolicySmall - Fast rollout policy for simulations
    3. ValueLarge - Position evaluation V(s)
    
    Uses PUCT formula for selection and mixes value net + rollout for evaluation.
    """
    
    def __init__(
        self,
        policy_large_checkpoint: Optional[str] = None,
        policy_small_checkpoint: Optional[str] = None,
        value_checkpoint: Optional[str] = None,
        num_simulations: int = ALPHAGO_NUM_SIMULATIONS,
        c_puct: float = ALPHAGO_C_PUCT,
        value_weight: float = ALPHAGO_VALUE_WEIGHT,
        max_depth: int = ALPHAGO_MAX_DEPTH,
        num_workers: int = ALPHAGO_NUM_WORKERS,
        device: Optional[str] = None
    ):
        """
        Initialize AlphaGo MCTS agent with three neural networks.
        
        Args:
            policy_large_checkpoint: Path to policy network (for priors)
            policy_small_checkpoint: Path to fast rollout policy
            value_checkpoint: Path to value network
            num_simulations: Number of MCTS simulations per move
            c_puct: PUCT exploration constant
            value_weight: λ for mixing value net and rollout (0-1)
            max_depth: Maximum rollout depth
            num_workers: Number of parallel workers (0 or 1 = sequential)
            device: Device for neural networks ("cpu", "cuda", "mps")
        """
        # Resolve checkpoint paths (relative to alpha_scoundrel/)
        alpha_scoundrel_dir = Path(__file__).parent.parent
        
        if policy_large_checkpoint is None:
            policy_large_checkpoint = alpha_scoundrel_dir / POLICY_LARGE_CHECKPOINT
        if policy_small_checkpoint is None:
            policy_small_checkpoint = alpha_scoundrel_dir / POLICY_SMALL_CHECKPOINT
        if value_checkpoint is None:
            value_checkpoint = alpha_scoundrel_dir / VALUE_LARGE_CHECKPOINT
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        
        # Load neural networks
        self.policy_large = PolicyLargeInference(
            checkpoint_path=policy_large_checkpoint,
            device=device
        )
        
        self.policy_small = PolicySmallInference(
            checkpoint_path=policy_small_checkpoint,
            device=device
        )
        
        self.value_net = ValueLargeInference(
            checkpoint_path=value_checkpoint,
            device=device
        )
        
        # MCTS parameters
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.value_weight = value_weight  # λ for mixing
        self.max_depth = max_depth
        self.num_workers = num_workers if num_workers > 1 else 0
        
        # Infrastructure
        self.translator = ScoundrelTranslator()
        self.transposition_table = TranspositionTable(ALPHAGO_TRANSPOSITION_TABLE_SIZE)
        
        # Statistics tracking
        self._last_root = None
        self._aggregated_cache_size = None
    
    def select_action(self, game_state: GameState) -> int:
        """
        Select best action using AlphaGo-style MCTS.
        
        Args:
            game_state: Current game state
            
        Returns:
            Action index (0-4)
        """
        if self.num_workers > 1:
            root = self._parallel_search(game_state)
        else:
            root = self._sequential_search(game_state)
        
        # Store root for statistics
        self._last_root = root
        
        # Return most visited child (same as vanilla MCTS)
        best_child = root.most_visited_child()
        return best_child.action
    
    def _sequential_search(self, game_state: GameState) -> AlphaGoNode:
        """
        Run MCTS simulations sequentially.
        
        4 Phases (AlphaGo-style):
        1. Selection - PUCT formula with policy priors
        2. Expansion - Add node and get policy priors
        3. Evaluation - Mix value network and rollout
        4. Backup - Propagate value up tree
        
        Args:
            game_state: Current game state
            
        Returns:
            Root node after search
        """
        root = self._create_node(game_state)
        
        # Get policy priors for root (critical for PUCT)
        self._add_policy_priors(root, game_state)
        
        for _ in range(self.num_simulations):
            # Determinize hidden information (shuffle unknown dungeon cards)
            sim_state = self._determinize_state(game_state)
            
            # 1. Selection - traverse tree using PUCT
            node, sim_state = self._select(root, sim_state)
            
            # 2. Expansion - add child if not terminal
            if not sim_state.game_over and not node.is_fully_expanded():
                node, sim_state = self._expand(node, sim_state)
            
            # 3. Evaluation - combine value net and rollout
            value = self._evaluate(node, sim_state)
            
            # 4. Backup - propagate value
            self._backpropagate(node, value)
        
        return root
    
    def _select(self, node: AlphaGoNode, game_state: GameState) -> Tuple[AlphaGoNode, GameState]:
        """
        Selection phase: Traverse tree using PUCT formula.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Key difference from vanilla MCTS:
        - Uses policy priors P(s,a) instead of uniform exploration
        - PUCT formula instead of UCB1
        
        Args:
            node: Starting node (usually root)
            game_state: Current game state
            
        Returns:
            Tuple of (selected_node, corresponding_game_state)
        """
        current_node = node
        current_state = game_state
        
        while current_node.is_fully_expanded() and current_node.children:
            if current_state.game_over:
                break
            
            # Use PUCT instead of UCB1
            current_node = current_node.best_child_puct(self.c_puct)
            action_enum = self.translator.decode_action(current_node.action)
            current_state = apply_action_to_state(current_state, action_enum)
        
        return current_node, current_state
    
    def _expand(self, node: AlphaGoNode, game_state: GameState) -> Tuple[AlphaGoNode, GameState]:
        """
        Expansion phase: Add child node and compute policy priors.
        
        Key difference from vanilla MCTS:
        - After adding child, get policy priors from PolicyLarge
        - Store priors in child node for PUCT selection
        
        Args:
            node: Node to expand
            game_state: Current game state
            
        Returns:
            Tuple of (new_child_node, new_game_state)
        """
        if game_state.game_over or len(node.untried_actions) == 0:
            return node, game_state
        
        # Select untried action (can be random or based on priors)
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Apply action
        action_enum = self.translator.decode_action(action)
        new_state = apply_action_to_state(game_state, action_enum)
        
        # Create child node
        child_node = self._create_node(new_state, parent=node, action=action)
        node.children.append(child_node)
        
        # Add policy priors for child (key AlphaGo enhancement)
        self._add_policy_priors(child_node, new_state)
        
        return child_node, new_state
    
    def _add_policy_priors(self, node: AlphaGoNode, game_state: GameState):
        """
        Get policy priors P(s,·) from PolicyLarge network.
        Store in node for PUCT selection.
        
        Args:
            node: Node to add priors to
            game_state: Current game state
        """
        if game_state.game_over:
            return
        
        # Get policy probabilities from neural network
        _, probs = self.policy_large(game_state)  # Returns (action, probs)
        node.set_prior_probs(probs)
    
    def _evaluate(self, node: AlphaGoNode, game_state: GameState) -> float:
        """
        Evaluation phase: Mix value network and rollout.
        
        V_final = (1 - λ) * V_value_net + λ * Z_rollout
        
        where λ = self.value_weight
        
        Key AlphaGo insight:
        - Value net provides global position evaluation
        - Rollout provides specific trajectory outcome
        - Mixing both gives robust evaluation
        
        Args:
            node: Node to evaluate
            game_state: Current game state
            
        Returns:
            Normalized value in [0, 1]
        """
        # Check transposition table first
        state_hash = self._hash_state(game_state)
        cached_value = self.transposition_table.get(state_hash)
        if cached_value is not None:
            return cached_value
        
        # Terminal state - use actual outcome
        if game_state.game_over:
            value = normalize_score(game_state.score)
            self.transposition_table.put(state_hash, value)
            return value
        
        # Non-terminal - mix value net and rollout
        
        # 1. Value network evaluation
        value_net_score = self.value_net(game_state)  # Raw score prediction
        value_net_normalized = normalize_score(value_net_score)
        
        # 2. Fast rollout evaluation
        rollout_value = self._rollout(game_state)
        
        # 3. Mix with λ
        final_value = (1 - self.value_weight) * value_net_normalized + self.value_weight * rollout_value
        
        # Cache result
        self.transposition_table.put(state_hash, final_value)
        
        return final_value
    
    def _rollout(self, game_state: GameState) -> float:
        """
        Fast rollout using PolicySmall network.
        
        Much faster than PolicyLarge (single FC layer vs 10 layers).
        Trades accuracy for speed in simulation phase.
        
        Args:
            game_state: Starting game state
            
        Returns:
            Normalized score from rollout
        """
        current_state = game_state
        depth = 0
        
        while depth < self.max_depth and not current_state.game_over:
            # Use fast policy network for action selection
            action_enum, _ = self.policy_small(current_state)
            current_state = apply_action_to_state(current_state, action_enum)
            depth += 1
        
        return normalize_score(current_state.score)
    
    def _backpropagate(self, node: AlphaGoNode, value: float):
        """
        Backpropagation phase: Update nodes with evaluation result.
        
        Same as vanilla MCTS - propagate value up the tree.
        
        Args:
            node: Starting node (leaf)
            value: Value to backpropagate
        """
        current_node = node
        while current_node is not None:
            current_node.update(value)  # visits += 1, value += value
            current_node = current_node.parent
    
    def _create_node(
        self,
        game_state: GameState,
        parent: Optional[AlphaGoNode] = None,
        action: Optional[int] = None
    ) -> AlphaGoNode:
        """
        Create AlphaGoNode from game state.
        
        Args:
            game_state: Game state
            parent: Parent node (None for root)
            action: Action taken to reach this node
            
        Returns:
            New AlphaGoNode
        """
        state_hash = self._hash_state(game_state)
        valid_actions = self._get_valid_actions(game_state)
        
        return AlphaGoNode(
            state_hash=state_hash,
            parent=parent,
            action=action,
            untried_actions=valid_actions,
            is_game_over=game_state.game_over
        )
    
    def _hash_state(self, game_state: GameState) -> int:
        """
        Create an integer hash representing the game state.
        
        Uses efficient integer-based hashing.
        Converts cards to integers: suit_index * 100 + card_value
        
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
        """
        Get list of valid action indices for current state.
        
        Args:
            game_state: Game state
            
        Returns:
            List of valid action indices
        """
        valid_actions = []
        
        for i in range(len(game_state.room)):
            valid_actions.append(i)
        
        if game_state.can_avoid:
            valid_actions.append(4)
        
        return valid_actions
    
    def _determinize_state(self, game_state: GameState) -> GameState:
        """
        Apply determinization to handle hidden information.
        Shuffles unknown cards in the dungeon while keeping known cards in place.
        
        Args:
            game_state: Game state
            
        Returns:
            Determinized copy of game state
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
        
        return determinized_state
    
    def _parallel_search(self, game_state: GameState) -> AlphaGoNode:
        """
        Parallel root parallelization.
        
        Each worker runs independent simulations, then aggregate results.
        
        Args:
            game_state: Current game state
            
        Returns:
            Aggregated root node
        """
        simulations_per_worker = self.num_simulations // self.num_workers
        
        # Create worker arguments
        worker_args = [
            (game_state, simulations_per_worker, self.c_puct, self.value_weight, 
             self.max_depth, self.device, str(self.policy_large.checkpoint_path),
             str(self.policy_small.checkpoint_path), str(self.value_net.checkpoint_path))
            for _ in range(self.num_workers)
        ]
        
        # Run parallel workers
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            worker_results = list(executor.map(_run_worker_search, worker_args))
        
        # Aggregate results
        root = self._aggregate_roots(worker_results, game_state)
        
        return root
    
    def _aggregate_roots(self, worker_results: List[Dict], game_state: GameState) -> AlphaGoNode:
        """
        Aggregate parallel worker results.
        
        Args:
            worker_results: List of worker result dictionaries
            game_state: Root game state
            
        Returns:
            Aggregated root node
        """
        # Create root node
        root = self._create_node(game_state)
        self._add_policy_priors(root, game_state)
        
        # Aggregate child statistics
        child_stats = {}  # action -> {visits, value}
        
        for result in worker_results:
            for action_idx, stats in result['child_stats'].items():
                if action_idx not in child_stats:
                    child_stats[action_idx] = {'visits': 0, 'value': 0.0}
                child_stats[action_idx]['visits'] += stats['visits']
                child_stats[action_idx]['value'] += stats['value']
        
        # Create children with aggregated stats
        for action_idx, stats in child_stats.items():
            action_enum = self.translator.decode_action(action_idx)
            child_state = apply_action_to_state(game_state, action_enum)
            
            child_node = self._create_node(child_state, parent=root, action=action_idx)
            child_node.visits = stats['visits']
            child_node.value = stats['value']
            
            self._add_policy_priors(child_node, child_state)
            root.children.append(child_node)
        
        # Update root visits
        root.visits = sum(stats['visits'] for stats in child_stats.values())
        
        # Aggregate cache stats
        self._aggregate_cache_stats(worker_results)
        
        return root
    
    def _aggregate_cache_stats(self, worker_results: List[Dict]):
        """
        Aggregate cache statistics from workers.
        
        Args:
            worker_results: List of worker result dictionaries
        """
        total_hits = sum(r['cache_stats']['hits'] for r in worker_results)
        total_misses = sum(r['cache_stats']['misses'] for r in worker_results)
        total_size = sum(r['cache_stats']['size'] for r in worker_results)
        
        self.transposition_table.hits = total_hits
        self.transposition_table.misses = total_misses
        self._aggregated_cache_size = total_size
    
    def get_action_stats(self) -> Dict[int, Dict]:
        """
        Get statistics for all actions (for viewer display).
        
        Returns:
            Dictionary mapping action_idx to {visits, mean_value, prior_prob}
        """
        if self._last_root is None:
            return {}
        
        stats = {}
        for child in self._last_root.children:
            mean_value = child.value / child.visits if child.visits > 0 else 0.0
            prior_prob = (self._last_root.prior_probs[child.action].item() 
                         if self._last_root.prior_probs is not None else 0.0)
            
            stats[child.action] = {
                'visits': child.visits,
                'mean_value': mean_value,
                'prior_prob': prior_prob
            }
        
        return stats
    
    def get_cache_stats(self) -> Dict:
        """
        Get transposition table statistics.
        
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


def _run_worker_search(args) -> Dict:
    """
    Worker function for parallel search.
    Runs independent MCTS simulations and returns results.
    
    Args:
        args: Tuple of (game_state, num_simulations, c_puct, value_weight, 
                        max_depth, device, policy_large_path, policy_small_path, value_path)
    
    Returns:
        Dictionary with child_stats and cache_stats
    """
    (game_state, num_simulations, c_puct, value_weight, max_depth, device,
     policy_large_path, policy_small_path, value_path) = args
    
    # Create worker agent
    agent = AlphaGoAgent(
        policy_large_checkpoint=policy_large_path,
        policy_small_checkpoint=policy_small_path,
        value_checkpoint=value_path,
        num_simulations=num_simulations,
        c_puct=c_puct,
        value_weight=value_weight,
        max_depth=max_depth,
        num_workers=0,  # No nested parallelization
        device=device
    )
    
    # Run search
    root = agent._sequential_search(game_state)
    
    # Extract child statistics
    child_stats = {}
    for child in root.children:
        child_stats[child.action] = {
            'visits': child.visits,
            'value': child.value
        }
    
    return {
        'child_stats': child_stats,
        'cache_stats': agent.transposition_table.stats()
    }


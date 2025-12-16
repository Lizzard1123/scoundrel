from scoundrel.game.game_manager import GameManager
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Play Scoundrel")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic deck shuffling (same seed = same game sequence)"
    )
    args = parser.parse_args()
    
    os.system('resize -s 14 88')
    game = GameManager(seed=args.seed)
    game.ui_loop()

if __name__ == "__main__":
    main()

from scoundrel.game.game_manager import GameManager
from scoundrel.models.card import CardType

def main():
    game = GameManager()

    print("Welcome to Scoundrel!")
    print("====================")

    while game.play_turn():
        pass

    # Game over
    print(f"\nGame Over! Final Score: {game.state.score}")


if __name__ == "__main__":
    main()

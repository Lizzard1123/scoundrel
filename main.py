from scoundrel.game.game_manager import GameManager
from scoundrel.models.card import CardType

def main():
    game = GameManager()

    print("Welcome to Scoundrel!")
    print("====================")

    while game.play_turn():
        pass

    # Game over
    final_health = game.state.health
    if final_health <= 0:
        # Calculate negative score
        remaining_monsters = [
            c
            for c in game.state.dungeon + game.state.room
            if c.type == CardType.MONSTER
        ]
        score = final_health - sum(m.value for m in remaining_monsters)
    else:
        # Calculate positive score
        last_potion = next(
            (c for c in reversed(game.state.discard) if c.type == CardType.POTION), None
        )
        score = final_health
        if final_health == 20 and last_potion:
            score += last_potion.value

    print(f"\nGame Over! Final Score: {score}")


if __name__ == "__main__":
    main()

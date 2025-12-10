from scoundrel.game.game_manager import GameManager
import os

def main():
    os.system('resize -s 14 88')
    game = GameManager()
    game.ui_loop()

if __name__ == "__main__":
    main()

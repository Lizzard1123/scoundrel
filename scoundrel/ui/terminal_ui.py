from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, CardType


class TerminalUI:
    def __init__(self):
        self.console = Console()

    def display_game_state(self, game_state: GameState):
        layout = Layout()
        layout.split_column(
            Layout(name="header"), Layout(name="game_area"), Layout(name="stats")
        )

        # Header
        header = Table.grid()
        header.add_row(
            Panel.fit(
                f"[bold]Scoundrel[/bold] - Health: {game_state.health}/20",
                style=(
                    "green"
                    if game_state.health > 10
                    else "yellow" if game_state.health > 5 else "red"
                ),
            )
        )

        # Game Area
        game_area = Table.grid()

        # Room
        room_display = Table.grid()
        room_display.add_row("[bold]Room:[/bold]")
        if game_state.room:
            room_cards = " ".join(str(card) for card in game_state.room)
            room_display.add_row(f"{room_cards}")

        # Weapon
        weapon_display = Table.grid()
        weapon_display.add_row("[bold]Equipped Weapon:[/bold]")
        if game_state.equipped_weapon:
            weapon_str = str(game_state.equipped_weapon)
            if game_state.weapon_monsters:
                monster_str = " ".join(str(m) for m in game_state.weapon_monsters)
                weapon_str += f" <- {monster_str}"
            weapon_display.add_row(weapon_str)

        game_area.add_row(room_display)
        game_area.add_row(weapon_display)

        # Stats
        stats = Table.grid()
        stats.add_row(f"Cards in Dungeon: {len(game_state.dungeon)}")
        stats.add_row(f"Cards in Discard: {len(game_state.discard)}")

        # Update layout
        layout["header"].update(header)
        layout["game_area"].update(game_area)
        layout["stats"].update(stats)

        self.console.print(layout)

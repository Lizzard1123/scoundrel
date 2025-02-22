from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.box import ROUNDED
from rich import box
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, CardType


class TerminalUI:
    def __init__(self):
        self.console = Console()

    def create_card_panel(self, card: Card, index: int = None) -> Panel:
        """Create a stylized panel for a card"""
        color = {
            CardType.MONSTER: "red",
            CardType.WEAPON: "yellow",
            CardType.POTION: "green",
        }[card.type]

        action = {
            CardType.MONSTER: "fight",
            CardType.WEAPON: "take",
            CardType.POTION: "heal",
        }[card.type]

        title = f"[{index}] {action}" if index is not None else ""

        content = f"[bold {color}]{str(card)}[/bold {color}]"
        if card.type == CardType.MONSTER:
            content += f" ({card.value}⚔️)"
        elif card.type == CardType.WEAPON:
            content += f" ({card.value}⚡)"
        elif card.type == CardType.POTION:
            content += f" ({card.value}❤️)"

        return Panel(
            content,
            title=title,
            border_style=color,
            box=box.ROUNDED,
            width=20,
            padding=(0, 1),
            height=3,
        )

    def create_weapon_stack(self, weapon: Card, monsters: list[Card]) -> str:
        """Create a compact weapon display"""
        if not weapon:
            return "[yellow]No weapon equipped[/yellow]"

        result = f"[yellow]Weapon: {str(weapon)}[/yellow] ({weapon.value}⚡)"
        if monsters:
            result += " [red]vs[/red] " + " ".join(
                f"[red]{str(m)}[/red]" for m in monsters
            )
        return result

    def display_game_state(self, game_state: GameState):
        """Display the game state in a compact, attractive layout at the bottom of the screen"""
        self.console.clear()

        # Get terminal height
        terminal_height = self.console.height

        # Calculate required height for our layout
        header_height = 4
        room_height = 5
        actions_height = 3
        total_height = header_height + room_height + actions_height

        # Print newlines to push content to bottom
        padding_lines = terminal_height - total_height - 1  # -1 for input prompt
        if padding_lines > 0:
            self.console.print("\n" * padding_lines, end="")

        # Create and print header
        health_color = (
            "green"
            if game_state.health > 10
            else "yellow" if game_state.health > 5 else "red"
        )
        header_content = (
            f"[bold {health_color}]Health: {game_state.health}/20[/bold {health_color}]  "
            f"[blue]Dungeon: {len(game_state.dungeon)}[/blue]  "
            f"[gray]Discard: {len(game_state.discard)}[/gray]\n"
            f"{self.create_weapon_stack(game_state.equipped_weapon, game_state.weapon_monsters)}"
        )

        header = Panel(
            header_content,
            title="[bold]Scoundrel[/bold]",
            box=ROUNDED,
            padding=(0, 1),
            height=header_height,
        )
        self.console.print(header)

        # Create and print room display
        room_cards = []
        if game_state.room:
            for i, card in enumerate(game_state.room, 1):
                room_cards.append(self.create_card_panel(card, i))

        room_display = Panel(
            Columns(room_cards, padding=1) if room_cards else "Empty room",
            title="[bold blue]Room[/bold blue]",
            border_style="blue",
            box=ROUNDED,
            padding=(0, 1),
            height=room_height,
        )
        self.console.print(room_display)

        # Create and print actions
        options = []
        if game_state.room:
            options.append("[bold white]avoid[/bold white]")
            for i, card in enumerate(game_state.room, 1):
                action = {
                    CardType.MONSTER: "fight",
                    CardType.WEAPON: "take",
                    CardType.POTION: "heal",
                }[card.type]
                options.append(f"[bold white]{action} {i}[/bold white]")

        if game_state.last_room_avoided:
            options.append("[dim red](Cannot avoid two rooms in a row)[/dim red]")

        actions = Panel(
            "Commands: " + ", ".join(options),
            title="[bold]Actions[/bold]",
            box=ROUNDED,
            padding=(0, 1),
            height=actions_height,
        )
        self.console.print(actions)

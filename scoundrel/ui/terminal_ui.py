from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align
from rich.box import ROUNDED
from rich import box
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, CardAction, CardColor, CardType


class TerminalUI:
    def __init__(self):
        self.console = Console()

    def create_card_panel(self, card: Card, index: int, game_state: GameState) -> Panel:
        """Create a stylized panel for a card"""
        color = CardColor[card.type]

        action = CardAction[card.type]

        title = f"[{index}]" if index is not None else ""

        content = f"{action}[bold {color}]{str(card)}[/bold {color}]"

        subtext = ""
        if not game_state.can_use_potion and card.type == CardType.POTION:
            subtext = "Discard"
        if game_state.equipped_weapon and card.type == CardType.WEAPON:
            subtext = "Replace"

        return Panel(
            content,
            title=title,
            subtitle=subtext,
            border_style=color,
            box=box.ROUNDED,
            width=20,
            padding=(0, 4),
            height=3,
        )

    def create_weapon_stack(self, weapon: Card | None, monsters: list[Card]) -> str:
        """Create a compact weapon display"""
        if not weapon:
            return "[yellow]No weapon equipped[/yellow]"

        result = f"[yellow]Weapon: {str(weapon)}[/yellow] "
        if monsters:
            result += "[red]vs[/red] " + " ".join(
                f"[red]{str(m)}[/red]" for m in monsters
            )
        return result

    def display_game_state(self, game_state: GameState):
        """Display the game state layout at the bottom of the screen"""
        self.console.clear()

        # Get terminal height
        terminal_height = self.console.height

        # Calculate required height for our layout
        header_height = 4
        room_height = 5
        actions_height = 3
        total_height = header_height + room_height + actions_height
        total_width = 88

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
            Align.center(header_content, vertical="middle"),
            title="[bold]Scoundrel[/bold]",
            box=ROUNDED,
            padding=(0, 1),
            width=total_width,
            height=header_height,
        )
        self.console.print(header)

        # Check if game over
        if(game_state.game_over):
            game_over_panel = Panel(
                Align.center(f"Score: {game_state.score}", vertical="middle"),
                title="[bold red]Game Over[/bold red]",
                box=ROUNDED,
                padding=(0, 1),
                width=total_width,
                height= room_height + actions_height,
            )
            self.console.print(game_over_panel)
            return

        # Create and print room display
        room_cards = []
        if game_state.room:
            for i, card in enumerate(game_state.room, 1):
                room_cards.append(self.create_card_panel(card, i, game_state))

        room_display = Panel(
            Columns(room_cards, padding=1, width=20) if room_cards else "Empty room",
            title="[bold blue]Room[/bold blue]",
            border_style="blue",
            box=ROUNDED,
            padding=(0, 1),
            width=total_width,
            height=room_height,
        )
        self.console.print(room_display)

        # Create and print actions
        options = []
        if game_state.room:
            if game_state.can_avoid:
                options.append("[bold white]avoid[/bold white]")
            for i, card in enumerate(game_state.room, 1):
                action = CardAction[card.type]
                options.append(f"[bold white]{action}{i}[/bold white]")

        actions = Panel(
            "Commands: " + ", ".join(options),
            title="[bold]Actions[/bold]",
            box=ROUNDED,
            padding=(0, 1),
            width=total_width,
            height=actions_height,
        )
        self.console.print(actions)

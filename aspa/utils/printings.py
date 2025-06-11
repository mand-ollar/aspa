import os


class Colors:
    """ANSI color codes"""

    BLACK: str = "\033[0;30m"
    RED: str = "\033[0;31m"
    GREEN: str = "\033[0;32m"
    BROWN: str = "\033[0;33m"
    BLUE: str = "\033[0;34m"
    PURPLE: str = "\033[0;35m"
    CYAN: str = "\033[0;36m"
    LIGHT_GRAY: str = "\033[0;37m"
    DARK_GRAY: str = "\033[0;30m"
    LIGHT_RED: str = "\033[0;31m"
    LIGHT_GREEN: str = "\033[0;32m"
    YELLOW: str = "\033[0;33m"
    LIGHT_BLUE: str = "\033[0;34m"
    LIGHT_PURPLE: str = "\033[0;35m"
    LIGHT_CYAN: str = "\033[0;36m"
    LIGHT_WHITE: str = "\033[0;37m"
    BOLD: str = "\033[1m"
    FAINT: str = "\033[2m"
    ITALIC: str = "\033[3m"
    UNDERLINE: str = "\033[4m"
    BLINK: str = "\033[5m"
    NEGATIVE: str = "\033[7m"
    CROSSED: str = "\033[9m"
    END: str = "\033[0m"


def clear_screen() -> None:
    os.system("stty sane; clear")


def clear_lines(num_lines: int) -> None:
    for _ in range(num_lines):
        print("\033[F\033[K", end="")

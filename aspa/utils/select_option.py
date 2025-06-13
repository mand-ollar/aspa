"""Select one of the multiple options from the list."""


def select_option(options: list[str], description: str = "Select option: ") -> int:
    """
    Args:
        options (list[str]): List of options with string type to choose from.
        description (str): Description of the options.
    """

    print(f"{description}\n")

    for i, option in enumerate(options):
        print(f"  - {i}] {option}")

    selected_option_idx_string = input("\nChoose option: ")
    print()

    while True:
        try:
            selected_option_idx_int = int(selected_option_idx_string)
            selected_option = options[selected_option_idx_int]
            print(f"You selected {selected_option}!")
            print()
            break

        except ValueError:
            print("Invalid input! Please enter a number.")
            selected_option_idx_string = input("Choose option: ")
            print()
            continue

        except IndexError:
            print("List out of range! Please try again.")
            selected_option_idx_string = input("Choose option: ")
            print()
            continue

    return selected_option_idx_int

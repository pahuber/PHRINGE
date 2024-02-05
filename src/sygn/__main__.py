"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """SYGN."""


if __name__ == "__main__":
    main(prog_name="sygn")  # pragma: no cover

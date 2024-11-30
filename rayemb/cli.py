# cli.py
import click
from .commands.generate import generate
from .commands.train import train
from .commands.evaluate import evaluate
from .commands.download import download

@click.group()
def cli():
    pass

cli.add_command(generate)
cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(download)

if __name__ == '__main__':
    cli()

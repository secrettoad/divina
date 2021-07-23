import sys
import click
from ..forecast import vision


@click.group()
def divina():
   pass


@click.argument('divina_version')
@click.argument('import_bucket')
@divina.command()
def forecast(divina_version, import_bucket):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


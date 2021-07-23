import sys
import click
from ..forecast import vision, build_dataset
import pkg_resources


@click.group()
def divina():
   pass


@click.argument('divina_version')
@click.argument('import_bucket')
@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


@divina.command()
def build_dataset():
    build_dataset.build_dataset()



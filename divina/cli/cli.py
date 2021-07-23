import sys
import click
from ..forecast import vision

@click.command()
def cli():
    mod = sys.argv[1]
    submods = sys.argv[2:]
    imported_mod = __import__('divina.{}'.format(mod), fromlist=['.'.join(submods)])
    import pdb
    pdb.set_trace()
    run = getattr(imported_mod, '.'.join(submods))
    # assuming your pattern has a run method defined.
    run()


@click.group()
def divina():
   pass


@click.argument('divina_version')
@click.argument('import_bucket')
@divina.command()
def forecast(divina_version, import_bucket):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


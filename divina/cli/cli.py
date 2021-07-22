import sys
import click


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

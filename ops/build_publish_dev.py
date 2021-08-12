import os
import subprocess
import sys
import pathlib
from pygit2 import Repository


def main():
    pkg_dir = pathlib.Path(*pathlib.Path().resolve().absolute().parts[:-1])
    branch = Repository(pkg_dir).head.shorthand
    commands = ['rm -rf {}'.format(os.path.join(pkg_dir, 'dist/*')),
                'cd {};python setup.py sdist bdist_wheel'.format(pkg_dir),
                'pip uninstall divina -y',
                'git commit -am "WIP"',
                'git push',
                'pip install git+https://github.com/tangentlabs/django-oscar-paypal.git@issue/34/oscar-0.6'.format(pkg_dir.resolve()),
                'rm -rf .eggs']
    run_commands(commands)


def run_commands(commands):
    my_env = os.environ.copy()
    my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
    for cmd in commands:

        process = subprocess.Popen(cmd, bufsize=1, universal_newlines=True, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=my_env)
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line),
                sys.stdout.flush()
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                sys.stderr.write(line),
                sys.stderr.flush()
        process.wait()
        exit_code = process.returncode
        if not exit_code == 0:
            sys.stdout.write('{}\n Could not upload package. Command failed with exit code {}\n'.format(cmd, exit_code))
    return


if __name__ == "__main__":
    main()

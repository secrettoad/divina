import os
import subprocess
import sys
import pathlib
from pkg_resources import get_distribution
import boto3


def main():
    pkg_dir = pathlib.Path(*pathlib.Path(__file__).parent.absolute().parts[:-2])
    auth_token = boto3.client('codeartifact').get_authorization_token(domain='coysu')['authorizationToken']
    commands = ['rm -rf {}'.format(os.path.join(pkg_dir, 'dist/*')),
                'aws codeartifact login --tool twine --repository divina --domain coysu',
                'cd {};python setup.py sdist bdist_wheel'.format(pkg_dir),
                'twine upload {} --repository=codeartifact'.format(os.path.join(pkg_dir, 'dist/*')),
                'python3 -m pip uninstall divina -y',
                'python3 -m pip install divina=={} -i https://aws:{}@coysu-169491045780.d.codeartifact.us-west-2.amazonaws.com/pypi/divina/simple/ --extra-index https://www.pypi.org/simple'.format(
                    get_distribution('divina').version, auth_token),
                'rm -rf .eggs']
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


if __name__=="__main__":
    main()

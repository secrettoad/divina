import os
import subprocess
import sys
import pathlib
import boto3


def main():
    pkg_dir = pathlib.Path(*pathlib.Path().resolve().absolute().parts[:-2])
    code_artifact_client = boto3.client('codeartifact')
    auth_token = code_artifact_client.get_authorization_token(domain='coysu')['authorizationToken']
    commands = ['rm -rf {}'.format(os.path.join(pkg_dir, 'dist/*')),
                'aws codeartifact login --tool twine --repository divina --domain coysu',
                'cd {};python setup.py sdist bdist_wheel'.format(pkg_dir),
                'twine upload {} --repository=codeartifact'.format(os.path.join(pkg_dir, 'dist/*')),
                'rm -rf .eggs']
    run_commands(commands)
    versions = code_artifact_client.list_package_versions(
        domain='coysu',
        repository='divina',
        format='pypi',
        package='divina',
        status='Published',
        sortBy='PUBLISHED_TIME'
    )
    commands = ['python3 -m pip install --upgrade divina[testing]=={} -i https://aws:{}@coysu-169491045780.d.codeartifact.us-west-2.amazonaws.com/pypi/divina/simple/ --extra-index https://www.pypi.org/simple'.format(
                    versions['versions'][0]['version'], auth_token)]
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

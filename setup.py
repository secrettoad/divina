"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup


setup(
    name='divina',  # Required
    version='0.0.1',  # Required
    author='Coysu Consulting',  # Optional
    author_email='john@coysu.com',  # Optional
    keywords=['divina', 'forecasting', 'coysu', 'causal', 'timeseries', 'aws'],  # Optional
    packages=['divina'],  # Required
    python_requires='>=3.6, <4',
    install_requires=['boto3==1.17.77', 'paramiko==2.7.2', 'backoff==1.10.0']
)
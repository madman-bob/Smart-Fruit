from setuptools import setup
from os import path

import re

project_root = path.join(path.abspath(path.dirname(__file__)), '..')


def get_version():
    with open(path.join(project_root, 'smart_fruit', '__init__.py'), encoding='utf-8') as init_file:
        return re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M).group(1)


def get_long_description():
    with open(path.join(project_root, 'README.rst'), encoding='utf-8') as readme_file:
        return readme_file.read()


def get_requirements():
    with open(path.join(project_root, 'requirements.txt'), encoding='utf-8') as requirements_file:
        return [requirement.strip() for requirement in requirements_file if requirement.strip()]


setup(
    name='smart-fruit',
    version=get_version(),
    packages=['smart_fruit'],
    install_requires=get_requirements(),

    author='Robert Wright',
    author_email='madman.bob@hotmail.co.uk',

    description='A Python schema-based machine learning library',
    long_description=get_long_description(),
    long_description_content_type='text/x-rst',
    url='https://github.com/madman-bob/Smart-Fruit',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.6'
)

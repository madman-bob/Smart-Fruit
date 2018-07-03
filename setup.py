from setuptools import setup
from os import path

import smart_fruit

here = path.abspath(path.dirname(__file__))


def get_long_description():
    with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
        return readme_file.read()


def get_requirements():
    with open(path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
        return [requirement.strip() for requirement in requirements_file if requirement.strip()]


setup(
    name='smart-fruit',
    version=smart_fruit.__version__,
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

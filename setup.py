from setuptools import find_packages, setup


def read_requirements(file):
    try:
        with open(file, encoding='utf-8') as f:
            return f.read().splitlines()
    except UnicodeError:
        with open(file, encoding='utf-16') as f:
            return f.read().splitlines()
        

requirements = read_requirements('requirements.txt')

setup(
    name='kaggle_arc_24',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.12.4'
)
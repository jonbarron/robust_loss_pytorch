import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


requirements = [
    'torch>=1.3.1',
    'torch-dct',
    'numpy>=1.15.4',
    'scipy>=1.1.0',
    'absl-py>=0.1.9',
    'mpmath>=1.1.0',
]

requirements_dev = [
    'tensorflow>=1.14.0',
    'tensorflow_probability>=0.7.0',
    'Pillow',
    'nose'
]


setup(
    name="robust_loss_pytorch",
    version="0.0.2",
    url="https://github.com/jonbarron/robust_loss_pytorch",
    license='Apache 2.0',
    author="Jonathan T. Barron",
    author_email="jonbarron@gmail.com",
    description="A General and Adaptive Robust Loss Function",
    long_description=read("README.md"),
    packages=find_packages(exclude=('tests',)),
    package_data={'': [
        '*.npz',
    ]},
    install_requires=requirements,
    extras_require={
        'dev': requirements_dev
    },
)

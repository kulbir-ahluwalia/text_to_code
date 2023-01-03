#from distutils.core import setup, find_packages
from setuptools import setup, find_packages


setup(
    name="text-to-code",
    version="1.0",
    description="Source code for text to code project",
    author="Rhaldar Michael Kulbir",
    packages=['seq_to_seq'],
    install_requires=[
        "jupyter",
        "torch",
        "transformers",
        "torchaudio",
        "torchvision",
        "jedi",
    ],
    package_data={'': ['list_of_input_output_sequences_only.json']},
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)

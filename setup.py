#from distutils.core import setup, find_packages
from setuptools import setup, find_packages


setup(
    name="text_to_code",
    version="1.0",
    description="Source code for text to code project",
    author="Rhaldar, Michael, Kulbir",
    packages=['text_to_code'],
    install_requires=[
        "jupyter==1.0.0",
        "torch",
        "transformers",
        "torchaudio",
        "torchvision",
        "gdown==4.5.1",
    ],
    package_data={'': ['list_of_input_output_sequences_only.json']},
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)

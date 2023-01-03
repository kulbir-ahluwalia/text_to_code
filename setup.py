#from distutils.core import setup, find_packages
from setuptools import setup, find_packages


setup(
    name="text_to_code",
    version="1.0",
    description="Source code for text to code project",
    author="Rhaldar, Michael, Kulbir",
    packages=['text_to_code'],
    install_requires=[
        "networkx==2.6.3",
        "h5py==3.7.0",
        "matplotlib==3.5.2",
        "numpy==1.21.6",
        "pillow",
        "pyyaml",
        "utm==0.7.0",
        "openai==0.20.0",
        "jupyter==1.0.0",
        "opencv-python==4.6.0.66",
        "torch",
        "transformers",
        "torchaudio",
        "torchvision",
        "ftfy==6.1.1",
        "regex==2022.6.2",
        "tqdm==4.64.0",
        # "clip @ git+https://github.com/openai/CLIP.git",
        "spacy==3.3.1",
        "gdown==4.5.1",
    ],
    package_data={'': ['./T5_sandbox/list_of_input_output_sequences_only.json']},
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)

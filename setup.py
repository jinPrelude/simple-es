from setuptools import setup

setup(
    name="simple_es",
    version="0.0.1",
    install_requires=[
        "matplotlib",
        "numpy",
        "wandb",
        "hydra-core",
        "torch",
        "gym",
        "gym[box2d]",
    ],
)

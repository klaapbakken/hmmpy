import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="hmmpy",
    version="0.1.0",
    description="Hidden Markov models in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/klaapbakken/hmmpy",
    author="Øyvind Klåpbakken",
    author_email="oyvind.klaapbakken@gmail.com",
    license="MIT",
    packages=["hmmpy"],
    include_package_data=True,
    install_requires=["numpy", "scipy", "tqdm"]
)
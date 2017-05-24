from setuptools import setup, find_packages

setup(
    name="celltk",
    version="0.1",
    packages=find_packages(),
    author='Takamasa Kudo',
    author_email='kudo@stanford.edu',
    license="MIT License",
    entry_points={
        "console_scripts": [
            "celltk=celltk.caller:main",
        ],
    }
)

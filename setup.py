from setuptools import setup, find_packages

setup(
    name="celltk",
    version="0.2",
    packages=find_packages(),
    author='Takamasa Kudo',
    author_email='kudo@stanford.edu',
    license="MIT License",
    entry_points={
        "console_scripts": [
            "celltk=celltk.caller:main",
            "celltk-preprocess=celltk.preprocess:main",
            "celltk-segment=celltk.segment:main",
            "celltk-subdetect=celltk.subdetect:main",
            "celltk-track=celltk.track:main",
            "celltk-postprocess=celltk.postprocess:main",
        ],
    }
)
